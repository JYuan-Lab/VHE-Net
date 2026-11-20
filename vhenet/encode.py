
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

GAP_CHARS = set("-. *\n\r\t")


_GENE_VOCAB = {"1": 5, "2": 6, "3": 7, "4": 8, "5": 9}
_DNA_TO_GENE = {"A": "1", "T": "2", "U": "2", "C": "3", "G": "4"}
_PAD_ID, _CLS_ID, _SEP_ID = 0, 2, 3


def clean_seq(seq: str) -> str:
    """Remove MSA gaps and invalid characters, and convert to uppercase."""
    return "".join(ch for ch in seq.upper() if ch not in GAP_CHARS)


def gene_seq_replace(seq: str) -> str:
    """Map DNA to the gene alphabet: A->'1', T/U->'2', C->'3', G->'4', other->'5'.

    Equivalent to LucaVirusTokenizer.gene_seq_replace, but called explicitly so
    the mapping does not depend on tokenizer behavior.
    """
    return "".join(_DNA_TO_GENE.get(ch.upper(), "5") for ch in seq)


def tokenize_gene_ids(seq: str, max_len: int = 1024):
    """Encode a DNA segment as LucaVirus gene-mode token ids.

    Returns (ids, mask) as Python lists of length ``max_len`` with layout
    [CLS] + gene ids + [SEP] + [PAD]*.
    """
    rep = gene_seq_replace(seq)[: max_len - 2]
    ids = [_CLS_ID] + [_GENE_VOCAB[c] for c in rep] + [_SEP_ID]
    ids = ids[:max_len] + [_PAD_ID] * max(0, max_len - len(ids))
    mask = [1 if i != _PAD_ID else 0 for i in ids]
    return ids, mask


def verify_gene_encoding(ids_cache_dir, fasta_path, n_virus: int = 3,
                         n_window: int = 3) -> bool:
    """Recompute sample windows and compare them with an existing id cache.

    Confirms that the cache was built with gene-mode ids rather than the
    tokenizer default. Returns True when every sampled window matches.
    """
    from vhenet.preprocessing import parse_fasta

    cache = Path(ids_cache_dir)
    index = json.loads((cache / "index.json").read_text())
    seq_dict = parse_fasta(fasta_path)
    names = [n for n in sorted(index) if n in seq_dict][:n_virus]
    total = ok = 0
    for name in names:
        data = torch.load(cache / index[name]["file"], map_location="cpu")
        ids, starts = data["ids"].long(), data["starts"]
        seq = clean_seq(seq_dict[name]) or "N"
        for w in range(min(n_window, len(starts))):
            s, e = starts[w]
            mine, _ = tokenize_gene_ids(seq[s:e], max_len=ids.shape[1])
            total += 1
            if mine == ids[w].tolist():
                ok += 1
    print(f"gene encoding check: {ok}/{total} (virus, window) pairs match exactly")
    return ok == total and total > 0


def window_starts(seq_len: int, window_size: int, stride: int):
    """Return (start, end) spans that cover the full sequence.

    Regular stride positions are 0, stride, 2*stride, ...; the final window is
    anchored so that its end equals seq_len and may overlap the previous one.
    """
    if seq_len <= window_size:
        return [(0, seq_len)]
    starts = list(range(0, seq_len - window_size + 1, stride))
    if starts[-1] + window_size < seq_len:
        starts.append(max(0, seq_len - window_size))
    return [(s, min(s + window_size, seq_len)) for s in starts]


def _encode_windows(seq: str, tokenizer, model, device,
                    window_size: int, stride: int,
                    encode_batch: int = 16):
    """Run LucaVirus over all windows of one sequence.

    Returns cls [Nwin, H], mean [Nwin, H], max [Nwin, H] (fp16), and the spans.
    """
    spans = window_starts(len(seq), window_size, stride)
    chunks = [seq[s:e] for s, e in spans]

    cls_list, mean_list, max_list = [], [], []
    n = len(chunks)
    for i in range(0, n, encode_batch):
        batch_text = chunks[i:i + encode_batch]
        # Apply gene mapping explicitly; the tokenizer would drop seq_type in
        # batch mode and fall back to the protein vocabulary.
        id_rows, mask_rows = [], []
        for text in batch_text:
            i_row, m_row = tokenize_gene_ids(text, max_len=window_size + 2)
            id_rows.append(i_row)
            mask_rows.append(m_row)
        ids = torch.tensor(id_rows, dtype=torch.long)
        mask = torch.tensor(mask_rows, dtype=torch.long)
        ids = ids.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state  # [b, L, H]
        mm = mask.to(hidden.dtype).unsqueeze(-1)
        cls = hidden[:, 0, :]
        mean = (hidden * mm).sum(dim=1) / mm.sum(dim=1).clamp_min(1.0)
        maxv = hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        cls_list.append(cls)
        mean_list.append(mean)
        max_list.append(maxv)
        del out, hidden

    return (torch.cat(cls_list, 0).float(),
            torch.cat(mean_list, 0).half(),
            torch.cat(max_list, 0).half(),
            spans)


def apply_whitening(cls: torch.Tensor, whiten_stats) -> torch.Tensor:
    """Whiten window CLS features with ``(cls - mean) @ W``.

    Whitening is the only step that removes the cross-virus collinear component
    and must be applied exactly once, while the CLS cache is built. On the
    209-virus reference cache, raw cross-virus cosine is 0.69-0.83 and drops to
    0.02-0.09 after whitening; un-whitening the cache with ``X @ W^{-1} + mean``
    restores a cosine of about 0.765.

    ``train.py`` and ``predict.py`` construct the model with
    ``whiten_stats=None`` because the shipped cache is already whitened; a
    second application would destroy the features. If this function returns raw
    CLS, pass the same statistics to the model.

    whiten_stats: `{"mean": [H], "W": [H, H]}` or None.
    """
    if whiten_stats is None:
        return cls
    mean = torch.as_tensor(whiten_stats["mean"], dtype=torch.float32)
    W = torch.as_tensor(whiten_stats["W"], dtype=torch.float32)
    if W.shape[0] != cls.shape[-1]:
        raise ValueError(
            f"whitening matrix shape {tuple(W.shape)} does not match CLS dimension {cls.shape[-1]}")
    cls = cls.float()

    mean = torch.as_tensor(
        whiten_stats["mean"],
        dtype=torch.float32,
        device=cls.device
    )
 
    W = torch.as_tensor(
        whiten_stats["W"],
        dtype=torch.float32,
        device=cls.device
    )
 
    return (cls - mean) @ W

def encode_fasta_to_cache(fasta_path: str, cache_dir: str,
                          tokenizer=None, model=None, device=None,
                          window_size: int = 1022, stride: int = 512,
                          encode_batch: int = 16, save_every: int = 50,
                          resume: bool = True, packed_path: str = None,
                          whiten_stats=None):
    """Encode window CLS features for an entire FASTA file.

    Two output formats are supported:

    * Directory format (default): one ``<idx>.pt`` per virus plus
      ``index.json``, consumed by ``load_window_cache()`` and
      ``FullWindowInteractionData``.
    * Packed single file (``packed_path='cache/cls_windows_cache_934.pt'``):
      a dict ``{virus: [Nwin, 2560] CLS}`` loaded directly by ``train.py`` and
      ``predict.py``. This is the format used by the reported runs.

    **Whitening:** when ``whiten_stats`` is provided, ``(cls - mean) @ W`` is
    applied before writing. The resulting cache is compatible with
    ``train.py`` / ``predict.py``, which construct the model with
    ``whiten_stats=None``. Without ``whiten_stats`` the raw CLS vectors are
    written, and the same statistics must be passed to the model; otherwise the
    feature distribution will differ from training. Do not apply both, or the
    features will be whitened twice.

    ``tokenizer`` and ``device`` remain only for API compatibility.
    Tokenization is performed explicitly by ``tokenize_gene_ids()``.

    Usage::

        from vhenet.encode import encode_fasta_to_cache
        stats = torch.load("data/whiten_stats.pt", map_location="cpu")
        encode_fasta_to_cache("data/virus_sequences.fasta",
                              "cache/cls_windows_cache_934",
                              model=model, device=dev,
                              whiten_stats=stats,          # Match the reported runs.
                              packed_path="cache/cls_windows_cache_934.pt")
    """
    from vhenet.preprocessing import parse_fasta
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"whitening: {'enabled (cls-mean)@W' if whiten_stats is not None else 'disabled (write raw CLS)'}")

    index_file = cache / "index.json"
    index = {}
    if resume and index_file.is_file():
        try:
            with open(index_file) as f:
                index = json.load(f)
            index = {k: v for k, v in index.items()
                     if (cache / v["file"]).is_file()}
            print(f"resume: {len(index)} viruses already complete; skipping them")
        except Exception:
            index = {}

    seq_dict = parse_fasta(fasta_path)
    names = sorted(seq_dict.keys())
    print(f"{len(names)} sequences to encode")

    packed = {} if packed_path else None
    model.eval()
    done = 0
    for idx, name in enumerate(names):
        if name in index:
            done += 1
            if packed is not None:
                d = torch.load(cache / index[name]["file"], map_location="cpu")
                packed[name] = d["cls"].float()
            continue
        seq = clean_seq(seq_dict[name])
        if not seq:
            seq = "N"
        try:
            cls, mean, maxv, spans = _encode_windows(
                seq, tokenizer, model, device, window_size, stride, encode_batch)
        except Exception as exc:  # A single failure should not abort the run.
            print(f"  [skip] {name}: {exc}")
            continue
        cls = apply_whitening(cls, whiten_stats)
        fn = cache / f"{idx:04d}.pt"
        torch.save({
            "cls": cls.cpu(),
            "mean": mean.cpu(),
            "max": maxv.cpu(),
            "starts": spans,
            "seq_len": len(seq),
        }, fn)
        if packed is not None:
            packed[name] = cls.float()
        index[name] = {"file": fn.name, "n_windows": cls.shape[0],
                       "seq_len": len(seq)}
        if (idx + 1) % save_every == 0 or idx == len(names) - 1:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=1)
            print(f"  ... {idx + 1}/{len(names)} saved index "
                  f"(newly done {idx + 1 - done})")

    with open(index_file, "w") as f:
        json.dump(index, f, indent=1)
    total = sum(v["n_windows"] for v in index.values())
    print(f"encoding complete: {len(index)} viruses, {total} windows -> {cache}")
    if packed is not None:
        # The packed file matches the directory format: whitened when
        # whiten_stats was supplied, raw otherwise. train.py / predict.py expect
        # a whitened cache, so pass whiten_stats when building it.
        torch.save(packed, packed_path)
        tag = "whitened" if whiten_stats is not None else "raw"
        print(f"packed file: {packed_path} ({len(packed)} viruses, {tag} CLS)")
    return index


def load_window_cache(cache_dir: str, max_viruses_in_ram: int = 200):
    """Lazy cache loader with an LRU of at most max_viruses_in_ram viruses."""
    cache = Path(cache_dir)
    with open(cache / "index.json") as f:
        index = json.load(f)
    ram: dict = {}
    order: list = []

    def get(name: str):
        if name in ram:
            order.remove(name)
            order.append(name)
            return ram[name]
        info = index.get(name)
        if info is None:
            raise KeyError(f"virus {name} not in cache index")
        data = torch.load(cache / info["file"], map_location="cpu")
        ram[name] = data
        order.append(name)
        if len(ram) > max_viruses_in_ram:
            old = order.pop(0)
            del ram[old]
        return data

    return get, index


# --------------------------------------------------------------------------- #
# CLI: rebuild the window CLS cache                                              #
# --------------------------------------------------------------------------- #
def verify_cache_whitening(cls_cache, whiten_stats, n_virus: int = 60,
                           verbose: bool = True):
    """Determine whether a window CLS cache is whitened or raw.

    Calibrated on the 209-virus reference cache:

    * Whitened: cross-virus CLS cosine is about 0.02-0.09 and per-dimension
      means are near zero.
    * Raw: cross-virus CLS cosine is about 0.69-0.83 because LucaVirus CLS
      features are strongly collinear.

    Un-whitening the cache with ``X @ W^{-1} + mean`` raises the cosine
    substantially for a whitened cache (about 0.037 -> 0.765) and lowers it for
    a raw cache.

    Parameters
    ----------
    cls_cache : str | Path | dict
        Path to the packed cache, or an already loaded
        ``{virus: [Nwin, 2560]}`` dictionary.
    whiten_stats : dict
        ``{"mean": [H], "W": [H, H]}``.

    Returns
    -------
    dict
        Includes ``is_whitened``, ``cosine_as_is``, ``cosine_unwhitened``, and
        ``mean_abs``.
    """
    data = (torch.load(str(cls_cache), map_location="cpu", weights_only=False)
            if isinstance(cls_cache, (str, Path)) else cls_cache)
    keys = sorted(data)[:n_virus]
    X = torch.cat([data[k].float() for k in keys], dim=0)
    if X.shape[0] < 2:
        raise ValueError("not enough samples: at least 2 window vectors are required")

    def cos(Z):
        V = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        C = V @ V.T
        n = C.shape[0]
        return C[~torch.eye(n, dtype=bool)].mean().item()

    mean = torch.as_tensor(whiten_stats["mean"], dtype=torch.float32)
    W = torch.as_tensor(whiten_stats["W"], dtype=torch.float32)
    c_raw = cos(X)
    X_un = X @ torch.linalg.pinv(W) + mean
    c_un = cos(X_un)
    mean_abs = X.mean(dim=0).abs().mean().item()

    # Whitened <=> low as-is cosine and a higher cosine after un-whitening.
    is_whitened = (c_raw < 0.35) and (c_un > c_raw)

    if verbose:
        print(f"cache whitening status ({len(keys)} viruses, {X.shape[0]} windows)")
        print(f"  as-is      cross-virus cosine = {c_raw:.4f}   |mean| = {mean_abs:.4f}")
        print(f"  unwhitened cross-virus cosine = {c_un:.4f}")
        print(f"  reference: whitened ~0.02-0.09 (cosine rises after un-whitening) | "
              f"raw ~0.69-0.83 (cosine falls after un-whitening)")
        print(f"  => {'whitened' if is_whitened else 'raw'}")
        if is_whitened:
            print("     consistent with whiten_stats=None in train.py / predict.py")
        else:
            print("     train.py / predict.py expect a whitened cache; pass "
                  "whiten_stats when building it, or pass the same statistics to the model")

    return {"is_whitened": is_whitened, "cosine_as_is": c_raw,
            "cosine_unwhitened": c_un, "mean_abs": mean_abs,
            "n_windows": int(X.shape[0])}


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Rebuild the LucaVirus window CLS cache (one .pt per virus plus index.json).")
    ap.add_argument("--fasta", default="data/virus_sequences.fasta",
                    help="input FASTA")
    ap.add_argument("--cache-dir", default="cache/cls_windows_cache_934",
                    help="output cache directory (one .pt per virus)")
    ap.add_argument("--model-path", default="pretrained/lucaVirus",
                    help="LucaVirus model directory")
    ap.add_argument("--window", type=int, default=1022, help="window length in bases")
    ap.add_argument("--stride", type=int, default=512, help="sliding-window stride")
    ap.add_argument("--batch", type=int, default=16, help="encoding batch size")
    ap.add_argument("--device", default=None,
                    help="cpu / cuda:0 ... (automatic by default)")
    ap.add_argument("--packed-path", default=None,
                    help="also write a packed dict{virus:[Nwin,2560]} file "
                         "consumed by train.py / predict.py")
    ap.add_argument("--whiten-stats", default="data/whiten_stats.pt",
                    help="whitening statistics .pt (mean/W). The reported cache is "
                         "whitened, so whitening is applied by default; pass 'none' for raw CLS")
    a = ap.parse_args()

    from transformers import AutoModel

    if a.device:
        device = torch.device(a.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    whiten_stats = None
    if a.whiten_stats and a.whiten_stats.lower() != "none":
        ws = Path(a.whiten_stats)
        if not ws.is_file():
            raise FileNotFoundError(
                f"whitening statistics not found: {ws} (pass --whiten-stats none to write raw CLS)")
        st = torch.load(ws, map_location="cpu", weights_only=False)
        whiten_stats = {"mean": st["mean"].float(), "W": st["W"].float()}
        print(f"whitening statistics: {ws}  mean{tuple(whiten_stats['mean'].shape)} "
              f"W{tuple(whiten_stats['W'].shape)}")

    print(f"loading LucaVirus: {a.model_path}")
    model = AutoModel.from_pretrained(a.model_path, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # The tokenizer is no longer used; gene mapping is explicit. Pass None only
    # to satisfy the function signature.
    encode_fasta_to_cache(
        fasta_path=a.fasta, cache_dir=a.cache_dir, tokenizer=None, model=model,
        device=device, window_size=a.window, stride=a.stride,
        encode_batch=a.batch, packed_path=a.packed_path,
        whiten_stats=whiten_stats)


if __name__ == "__main__":
    _main()

"""Cache token ids for every sequence window (lossless representation).

Diagnostic notes that distinguish the raw and whitened feature spaces:

  * Raw CLS vectors are strongly collinear across viruses: mean pairwise cosine
    0.69 on the 934-virus training set and 0.83 on a 209-virus external set;
    windows from the same virus also reach 0.82.
  * Whitening, ``whiten = (cls - mean) @ W``, reduces the mean pairwise cosine
    to 0.02 on the 934 set and 0.09 on the 209 set. Quote the value together
    with the feature space it refers to.
  * Mean pooling is permutation invariant: ``(1/L) * sum(h(token_i))`` does not
    depend on token order, so shuffling a region cannot change the mean-pooled
    representation. Only CLS can carry order information.
  * Individual nucleotide embeddings have low discriminability (cosine
    0.13-0.21 across A/C/G/T).

This module therefore caches token ids only and leaves content modeling to the
frozen LucaVirus nucleotide embedding table and trainable downstream layers.

Gene-mode caveat: the LucaVirus tokenizer respects ``seq_type`` only on the
single-string path. Passing a list goes through ``batch_encode_plus``, which
drops ``seq_type`` (see tokenization_lucavirus.py:294), so DNA is looked up in
the protein vocabulary. No exception is raised, but every id is wrong. This
module avoids the tokenizer and calls
``vhenet.encode.tokenize_gene_ids()`` explicitly; use
``vhenet.encode.verify_gene_encoding()`` to validate an existing cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vhenet.encode import clean_seq, tokenize_gene_ids, window_starts

# Tokenizer id -> base code (A=0, T=1, G=2, C=3, other=4).
_TOKEN_ID_MAP = {5: 0, 6: 1, 8: 2, 7: 3}


def _build_kmer_hist(ids: torch.Tensor, k: int = 6, n_bins: int = 4096):
    """ids: [Nwin, L] int16 -> [Nwin, n_bins] float counts.

    Uses a rolling 6-mer hash, ``code = sum(base_i * 4^i)``, and stores raw
    counts per window.
    """
    import numpy as np
    ids_np = ids.numpy()
    base = np.zeros_like(ids_np, dtype=np.int64)
    for tid, b in _TOKEN_ID_MAP.items():
        base[ids_np == tid] = b
    # Map invalid tokens, including special tokens, to base code 4.
    base[~np.isin(ids_np, list(_TOKEN_ID_MAP.keys()))] = 4
    Nwin, L = base.shape
    codes = np.zeros((Nwin, max(0, L - k + 1)), dtype=np.int64)
    p4 = 1
    for i in range(k):
        codes += base[:, i:i + L - k + 1] * p4
        p4 *= 4
    hist = np.zeros((Nwin, n_bins), dtype=np.float32)
    for w in range(Nwin):
        row = codes[w]
        row = row[row < n_bins]  # Drop k-mers containing N.
        if row.size:
            counts = np.bincount(row, minlength=n_bins)
            hist[w] = counts.astype(np.float32)  # Raw counts; the model applies log1p.
    return hist


def encode_ids_to_cache(fasta_path: str, cache_dir: str, tokenizer=None,
                        window_size: int = 1022, stride: int = 512,
                        save_every: int = 100, resume: bool = True,
                        kmer: int = 6):
    """Cache token ids and k-mer spectra for every window in a FASTA file.

    Each virus gets one file::

        {'ids': [Nwin, L] int16, 'mask': [Nwin, L] bool,
         'hist': [Nwin, 4^k] float32, 'starts': [[s, e], ...],
         'seq_len': int}

    The ``tokenizer`` argument is retained only for API compatibility and is
    unused; tokenization is performed explicitly by
    ``vhenet.encode.tokenize_gene_ids`` so that ``seq_type`` cannot be silently
    dropped.

    Usage::

        from vhenet.encode_ids import encode_ids_to_cache
        encode_ids_to_cache("data/virus_sequences.fasta",
                            "cache/ids_km_cache_934", kmer=6)
    """
    from vhenet.preprocessing import parse_fasta
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / "index.json"
    index = {}
    if resume and index_file.is_file():
        try:
            with open(index_file) as f:
                index = json.load(f)
            index = {k: v for k, v in index.items()
                     if (cache / v["file"]).is_file()}
        except Exception:
            index = {}

    seq_dict = parse_fasta(fasta_path)
    names = sorted(seq_dict.keys())
    L = window_size + 2  # CLS + SEQ + SEP
    print(f"{len(names)} sequences to tokenize")

    for idx, name in enumerate(names):
        if name in index:
            continue
        seq = clean_seq(seq_dict[name]) or "N"
        spans = window_starts(len(seq), window_size, stride)
        chunks = [seq[s:e] for s, e in spans]
        # Apply gene mapping explicitly instead of going through the tokenizer:
        # batch tokenization silently drops seq_type, falls back to the protein
        # vocabulary, and treats DNA characters as amino acids. See the module
        # docstring and encode.py for details.
        id_rows, mask_rows = [], []
        for text in chunks:
            i_row, m_row = tokenize_gene_ids(text, max_len=L)
            id_rows.append(i_row)
            mask_rows.append(m_row)
        ids = torch.tensor(id_rows, dtype=torch.int16)    # [Nwin, L]
        mask = torch.tensor(mask_rows, dtype=torch.bool)  # [Nwin, L]
        hist = None
        if kmer and kmer > 0:
            hist = torch.from_numpy(
                _build_kmer_hist(ids, k=kmer, n_bins=4 ** kmer))
        fn = cache / f"{idx:04d}.pt"
        torch.save({"ids": ids, "mask": mask, "hist": hist, "starts": spans,
                    "seq_len": len(seq)}, fn)
        index[name] = {"file": fn.name, "n_windows": ids.shape[0],
                       "seq_len": len(seq)}
        if (idx + 1) % save_every == 0 or idx == len(names) - 1:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=1)
            print(f"  ... {idx + 1}/{len(names)} saved index")

    with open(index_file, "w") as f:
        json.dump(index, f, indent=1)
    print(f"tokenization complete: {len(index)} viruses -> {cache}")

    # ---- Corpus-level z-score statistics over log1p(raw counts) ----
    if kmer and kmer > 0:
        stats_file = cache / "hist_stats.pt"
        if not stats_file.is_file():
            print("computing k-mer spectrum statistics ...")
            import torch as _t
            sums = None
            sumsq = None
            total_windows = 0
            for info in index.values():
                d = _t.load(cache / info["file"], map_location="cpu")
                h = _t.as_tensor(d["hist"], dtype=_t.float32)
                h = _t.log1p(h)
                s = h.sum(dim=0)
                q = (h * h).sum(dim=0)
                sums = s if sums is None else sums + s
                sumsq = q if sumsq is None else sumsq + q
                total_windows += h.shape[0]
            mean = sums / total_windows
            var = sumsq / total_windows - mean * mean
            std = var.clamp_min(1e-4).sqrt()
            _t.save({"mean": mean, "std": std}, stats_file)
            print(f"saved hist_stats: {stats_file} (windows={total_windows})")
    return index


def load_ids_cache(cache_dir: str, max_viruses_in_ram: int = 500):
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
# CLI: rebuild the token-id cache                                               #
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Rebuild the k-mer token-id cache (one .pt per virus plus index.json).")
    ap.add_argument("--fasta", default="data/virus_sequences.fasta")
    ap.add_argument("--cache-dir", default="cache/ids_km_cache_934")
    ap.add_argument("--window", type=int, default=1022)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--kmer", type=int, default=6,
                    help="k for the k-mer spectrum (0 disables hist)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing index.json and rebuild from scratch")
    a = ap.parse_args()

    encode_ids_to_cache(
        fasta_path=a.fasta, cache_dir=a.cache_dir,
        window_size=a.window, stride=a.stride, kmer=a.kmer,
        resume=not a.no_resume)


if __name__ == "__main__":
    _main()

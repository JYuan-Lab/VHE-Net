#!/usr/bin/env python3

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Reference values produced with torch 2.5.1+cu118 / transformers 4.36.2
EXPECT = {
    "Bovine atadenovirus E": (58, "b7108a4b6edbedac5bacca3a45ba50d077a2d54f66fcdcd96f7778fa0042e938"),
    "Rhinovirus C": (13, "2eb4eff4434791bdf6dc23dc9a042b105c3002a99dbfa160acfea67ba16d2ac0"),
}


def extract(fasta: Path, names, out: Path) -> None:
    keep, hit = [], False
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                hit = line[1:].split(" Accessions")[0].strip() in names
            if hit:
                keep.append(line)
    out.write_text("".join(keep))


def payload_sha(t: torch.Tensor) -> str:
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fasta", default="data/virus_sequences.fasta")
    ap.add_argument("--model-path", default="pretrained/lucaVirus")
    ap.add_argument("--whiten-stats", default="data/whiten_stats.pt")
    a = ap.parse_args()

    fasta = (root / a.fasta).resolve()
    if not fasta.is_file():
        print(f"FASTA file not found: {fasta}")
        return 1

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__} | cuda {torch.version.cuda}"
          f" | cudnn {torch.backends.cudnn.version()}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="vhenet_check_") as td:
        td = Path(td)
        two = td / "two.fa"
        extract(fasta, set(EXPECT), two)
        n = sum(1 for line in two.open() if line.startswith(">"))
        if n != len(EXPECT):
            print(f"Found only {n}/{len(EXPECT)} reference sequences in the FASTA; names do not match")
            return 1

        cmd = [
            sys.executable, "-m", "vhenet.encode",
            "--fasta", str(two),
            "--cache-dir", str(td / "cache"),
            "--packed-path", str(td / "two.pt"),
            "--model-path", str((root / a.model_path).resolve()),
            "--whiten-stats", str((root / a.whiten_stats).resolve()),
            "--device", a.device,
        ]
        env = {**os.environ, "PYTHONPATH": str(root)}
        r = subprocess.run(cmd, cwd=root, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return r.returncode

        packed = torch.load(td / "two.pt", map_location="cpu", weights_only=False)

    bad = 0
    print(f"{'Virus':26s} {'Windows':>8s} {'dtype':>9s} {'payload SHA-256':>20s}  Status")
    for name, (nwin, want) in EXPECT.items():
        t = packed.get(name)
        if t is None:
            print(f"{name[:26]:26s} {'-':>8s} {'-':>9s} {'missing':>20s}  FAIL")
            bad += 1
            continue
        got = payload_sha(t)
        ok = (t.shape[0] == nwin) and (t.dtype == torch.float32) and (got == want)
        bad += 0 if ok else 1
        print(f"{name[:26]:26s} {t.shape[0]:6d} {str(t.dtype).replace('torch.', ''):>9s}"
              f" {got[:8] + '…' + got[-8:]:>20s}  {'OK' if ok else 'FAIL'}")

    print()
    if bad:
        print(f"{bad} reference sequence(s) failed: the torch/transformers versions are not "
              "2.5.1+cu118 / 4.36.2, or the encoder implementation differs from the "
              "released version")
        return 1
    print("PASS: the environment reproduces the released cache bit-for-bit; the full cache can be rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# cache/

Precomputed feature caches. These files are **not tracked by Git** (about 1 GB).

## Required contents

```text
cache/
|-- ids_km_cache_934/          # token ids + k-mer spectra (936 files, ~654 MB)
`-- cls_windows_cache_934.pt   # window CLS features (934 viruses x windows x 2560, ~360 MB)
```

The two formats serve different purposes:

| Path | Format | Used by |
|---|---|---|
| `cache/ids_km_cache_934/` | Directory: one `.pt` per virus plus `index.json` | `train.py` / `predict.py` through `FullWindowIdsHistPosData` and `load_ids_cache` |
| `cache/cls_windows_cache_934.pt` | Single `.pt` file: `dict{virus: Tensor[Nwin, 2560]}` | loaded directly by `train.py` and `predict.py` |

## Why caches are needed

LucaVirus has 0.95 B parameters. Re-encoding 934 viral genomes on every epoch
would dominate runtime. With the precomputed caches, training reads only the
stored features and the encoder never enters the autograd graph.

## Rebuilding the caches

```bash
# 1) Token ids and k-mer spectra (CPU only)
python -m vhenet.encode_ids \
    --fasta data/virus_sequences.fasta \
    --cache-dir cache/ids_km_cache_934 \
    --kmer 6

# 2) Window CLS features (requires a GPU and LucaVirus weights)
python -m vhenet.encode \
    --fasta data/virus_sequences.fasta \
    --model-path pretrained/lucaVirus \
    --cache-dir cache/cls_windows_cache_934 \
    --packed-path cache/cls_windows_cache_934.pt \
    --window 1022 --stride 512 \
    --whiten-stats data/whiten_stats.pt
```

The CLS cache stores **already-whitened** vectors. `train.py` and `predict.py`
construct the model with `whiten_stats=None`, so whitening must be applied once
while building this cache, as shown above. Do not whiten the features a second
time.

To validate a rebuilt cache:

```python
import torch
from vhenet.encode import verify_cache_whitening, verify_gene_encoding

stats = torch.load("data/whiten_stats.pt", map_location="cpu")
verify_cache_whitening("cache/cls_windows_cache_934.pt", stats, n_virus=60)
verify_gene_encoding("cache/ids_km_cache_934", "data/virus_sequences.fasta")
```

The tokenizer must not be used in batch mode for gene sequences: batch encoding
silently drops `seq_type` and falls back to the protein vocabulary. Both cache
builders call `vhenet.encode.tokenize_gene_ids()` explicitly instead.

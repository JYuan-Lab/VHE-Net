# checkpoints/

Trained model weights. These files are **not tracked by Git**. Each checkpoint is
about 3.6 GB because it includes the frozen LucaVirus encoder.

The provided configuration writes:

```text
checkpoints/
`-- vhe_net_with/
    `-- last_model.pt
```


## Rebuilding the checkpoints

```bash
python train.py --config configs/vhe_net_with_weight.yaml
```

The default configuration uses importance-weighted training. For an unweighted
control, set `data.imp_upper: 1.0` and use a separate
`training.model_save_path`, such as `checkpoints/vhe_net_without`.

Obtaining pre-trained weights is described in the root `README.md`.

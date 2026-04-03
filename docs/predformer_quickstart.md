# PredFormer Quickstart

Use `predformer` as `model.name` in YAML.

Starter config:

- `configs/full/predformer_2010_2013_nospeedopt.yaml`

Run training:

```bash
python -m ocean_forecast.train --config configs/full/predformer_2010_2013_nospeedopt.yaml
```

Model keys for `predformer`:

- `embed_dim`
- `num_layers`
- `num_heads`
- `mlp_ratio`
- `dropout`
- `attn_dropout`
- `patch_size`
- `temporal_kernel` (`binary_ts` or `full`)
- `max_input_len`
- `max_pred_len`

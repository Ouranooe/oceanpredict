# Yellow-Bohai Sea Forecasting Baselines

This repository now includes a modular baseline for:

- Input: historical `SST/SSS/SSU/SSV`
- Output: next 72-hour `SST/SSS/speed`
- Area: Yellow-Bohai Sea (`31-41N`, `117-127E`)
- Data source: `dataset/海域要素预测` only

## Structure

- `ocean_forecast/data`: zip NetCDF reader, split/window logic, dataset, normalization stats
- `ocean_forecast/models`: base interface, ConvLSTM, TAU, CNN+Transformer models, registry
- `ocean_forecast/train.py`: training + dev early stop + test evaluation
- `ocean_forecast/infer.py`: forecast inference from a start time
- `configs/convlstm_baseline.yaml`: default config
- `configs/tau_baseline.yaml`: TAU config
- `configs/cnn_transformer_baseline.yaml`: CNN Encoder + Transformer Backbone + CNN Decoder
- `configs/cnn_transformer_physics.yaml`: Transformer + physics loss regularization

## Train

```bash
python train.py --config configs/convlstm_baseline.yaml
```

Alternative models:

```bash
python train.py --config configs/tau_baseline.yaml
python train.py --config configs/cnn_transformer_baseline.yaml
python train.py --config configs/cnn_transformer_physics.yaml
```

Debug smoke:

```bash
python train.py --config configs/convlstm_baseline.yaml --output_dir outputs/smoke --max_train_batches 2 --max_eval_batches 2
```

## Infer

```bash
python infer.py \
  --config configs/convlstm_mid_8gb.yaml \
  --ckpt outputs/train_mid_8gb_full/best.ckpt \
  --start_time 2014-06-01T00:00:00 \
  --viz_dir outputs/convlstm_baseline/viz_20140601 \
  --metrics_json outputs/convlstm_baseline/viz_20140601/metrics.json \
  --sample_every_hours 9 \
  --auto_shift_start
```

## Notes

- The code is path-agnostic and runs in Linux/WSL.
- Device selection supports `auto/cpu/cuda`.
- Inference now saves visualization PNGs (no npz/csv output).
- If forecast ground truth exists, inference also reports RMSE/NRMSE (`NRMSE = RMSE / train_std`) and writes metrics JSON.

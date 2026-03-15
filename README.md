# Yellow-Bohai Sea ConvLSTM Baseline

This repository now includes a modular baseline for:

- Input: historical `SST/SSS/SSU/SSV`
- Output: next 72-hour `SST/SSS/speed`
- Area: Yellow-Bohai Sea (`31-41N`, `117-127E`)
- Data source: `dataset/海域要素预测` only

## Structure

- `ocean_forecast/data`: zip NetCDF reader, split/window logic, dataset, normalization stats
- `ocean_forecast/models`: base interface, ConvLSTM model, registry
- `ocean_forecast/train.py`: training + dev early stop + test evaluation
- `ocean_forecast/infer.py`: forecast inference from a start time
- `configs/convlstm_baseline.yaml`: default config

## Train

```bash
python train.py --config configs/convlstm_baseline.yaml
```

Debug smoke:

```bash
python train.py --config configs/convlstm_baseline.yaml --output_dir outputs/smoke --max_train_batches 2 --max_eval_batches 2
```

## Infer

```bash
python infer.py \
  --config configs/convlstm_baseline.yaml \
  --ckpt outputs/convlstm_baseline/best.ckpt \
  --start_time 2014-06-01T00:00:00 \
  --output outputs/convlstm_baseline/pred_20140601.npz \
  --stats_csv outputs/convlstm_baseline/pred_20140601_stats.csv \
  --auto_shift_start
```

## Notes

- The code is path-agnostic and runs in Linux/WSL.
- Device selection supports `auto/cpu/cuda`.
- NRMSE is computed as `RMSE / train_std` per channel, then averaged across `sst/sss/speed`.

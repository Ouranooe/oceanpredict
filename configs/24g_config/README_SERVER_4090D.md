# 4090D 24GB Server Quickstart

## Updated dataset paths
All 4 compare configs now use:

- `data.base_root: /root/autodl-tmp/dataset_base/sea_v1`
- `data.root_dir: /root/autodl-tmp/dataset_base/sea_v1`

Files:

- `configs/24g_config/convlstm_4090d_24g_compare.yaml`
- `configs/24g_config/tau_4090d_24g_compare.yaml`
- `configs/24g_config/cnn_transformer_4090d_24g_compare.yaml`
- `configs/24g_config/cnn_transformer_physics_4090d_24g_compare.yaml`

## One-command Conda install
An environment file is added at repo root:

- `environment.server.4090d.yml`

Run in project root:

```bash
conda env remove -n ocean4090d -y >/dev/null 2>&1 || true && conda env create -f environment.server.4090d.yml
```

Activate:

```bash
conda activate ocean4090d
```

Optional CUDA check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Train commands

```bash
python train.py --config configs/24g_config/convlstm_4090d_24g_compare.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_96_72.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_96_96.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_120_72.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_120_96.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_144_72.yaml
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_144_96.yaml
python train.py --config configs/24g_config/tau_4090d_24g_compare.yaml
python train.py --config configs/24g_config/cnn_transformer_4090d_24g_compare.yaml
python train.py --config configs/24g_config/cnn_transformer_physics_4090d_24g_compare.yaml


python train.py --config configs/24g_config/convlstm_4090d_24g_compare_168_72.yaml && \
python train.py --config configs/24g_config/convlstm_4090d_24g_compare_192_72.yaml

```

## Notes
- These configs already use the new `density_physics_loss` + `smoothness_loss` naming.
- `diffdiv_loss` and `laplacian_smoothness_loss` remain in config but are not included in the current v1 main loss path.

python scripts/eval_final_from_best.py --config configs/24g_config/convlstm_4090d_24g_compare_168_72.yaml
python scripts/eval_final_from_best.py --config configs/24g_config/convlstm_4090d_24g_compare_192_72.yaml

python train.py --config configs/24g_config/tau_4090d_24g_compare.yaml
python train.py --config configs/24g_config/cnn_transformer_4090d_24g_compare.yaml
python train.py --config configs/24g_config/cnn_transformer_physics_4090d_24g_compare.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full/cnn_transformer_4090d_24g_compare.yaml && \
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full/convlstm_4090d_24g_compare_120_72.yaml

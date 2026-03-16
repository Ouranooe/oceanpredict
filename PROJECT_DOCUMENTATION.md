# oceanSys 项目文档（基于当前仓库实况）

## 1. 项目定位

本项目是一个面向黄渤海海域的时空序列预测基线系统，核心目标是：

- 输入：历史 72 小时 `SST/SSS/SSU/SSV`
- 输出：未来 72 小时 `SST/SSS/speed`
- 模型：ConvLSTM（双层编码器 + 双层解码器）
- 训练范式：监督学习 + 掩码 MSE + 按通道 RMSE/NRMSE 评估

入口脚本为仓库根目录：

- `train.py`（调用 `ocean_forecast.train.main`）
- `infer.py`（调用 `ocean_forecast.infer.main`）

## 2. 仓库结构总览

```text
system/
├─ configs/                      # 训练配置（baseline / mid_8gb）
├─ dataset/                      # 原始压缩数据
│  ├─ 海域要素预测/               # 1994-2014 年度 zip（本项目主数据）
│  └─ 风浪异常识别/               # 2016-2024 月度 zip（当前主流程未使用）
├─ ocean_forecast/               # 主包
│  ├─ data/                      # 读取 zip NetCDF、切分窗口、Dataset、统计量
│  ├─ models/                    # BaseForecaster + ConvLSTM + 注册器
│  ├─ train.py                   # 训练与评估主流程
│  ├─ infer.py                   # 推理主流程
│  ├─ losses.py                  # masked MSE
│  ├─ metrics.py                 # RMSE/NRMSE 累计与汇总
│  ├─ config.py                  # 默认配置、YAML 合并与保存
│  └─ utils.py                   # 随机种子与设备解析
├─ scripts/
│  └─ smoke_run.py               # 一键 smoke（小训练 + 推理）
├─ tests/                        # 读取/窗口/模型/指标单测
├─ outputs/                      # 训练与推理产物样例
├─ build_numpy_data.py           # 独立数据构建脚本（不在主训练链路）
└─ README.md
```

## 3. 技术栈与依赖

从源码导入可见，主流程依赖包括：

- `python`（项目内存在 `cpython-311` 与 `cpython-313` 缓存，说明至少在两个版本运行过）
- `torch`
- `numpy`
- `h5py`
- `pyyaml`

测试与辅助脚本额外使用：

- `pytest`
- `xarray`（`build_numpy_data.py`）
- `pandas`（`build_numpy_data.py`）

仓库中当前未看到 `requirements.txt` / `pyproject.toml`，依赖安装需要按上面模块手动准备。

## 4. 数据资产说明

### 4.1 主训练数据（已接入主流程）

- 目录：`dataset/海域要素预测`
- 文件形态：按年压缩包（`1994.zip` ... `2014.zip`）
- 数量：21 个 zip
- 体量：约 9.18 GiB

### 4.2 其他数据（当前主流程未使用）

- 目录：`dataset/风浪异常识别`
- 文件形态：按年月压缩包（2016-2024）
- 数量：108 个 zip
- 体量：约 10.19 GiB

### 4.3 读取方式

`ZipNetCDFReader` 直接从 zip 内读取 `.nc`（不落盘），并完成：

- `time` 四舍五入到小时索引
- `_FillValue/scale_factor/add_offset` 解码
- 去重同小时帧
- LRU 成员缓存（`cache_size` 可配）

## 5. 训练/验证/测试切分与样本构造

切分字段在 `data` 配置中定义：

- 训练主段：`train_start` 到 `train_main_end`
- 验证段：`dev_start` 到 `dev_end`
- 测试段：`test_start` 到 `test_end`

窗口规则：

- 单样本总长度 = `input_len + pred_len`
- 只有时间连续（逐小时）窗口会被保留
- `stride` 控制窗口起点步长

归一化统计：

- 基于训练主段逐帧统计 `input_mean/std`、`target_mean/std`
- `speed = sqrt(ssu^2 + ssv^2)`
- 保存为 `stats.npz`

## 6. 模型与损失评估

### 6.1 模型（ConvLSTM）

- 输入：`[B, Tin, 4, H, W]`
- 输出：`[B, Tout, 3, H, W]`
- 结构：
  - 编码器 ConvLSTMCell × 2
  - 解码器 ConvLSTMCell × 2（以零场自回归展开）
  - `1x1 Conv` 输出头

### 6.2 损失

- `masked_mse_loss` 仅在海洋网格掩码区域计算误差

### 6.3 指标

- `rmse_sst / rmse_sss / rmse_speed`
- `nrmse_* = rmse / train_std`
- `nrmse_mean` 为三通道均值

## 7. 训练流程

训练主流程（`ocean_forecast/train.py`）包含：

1. 读取并合并 YAML 配置
2. 建立数据索引与时间切分
3. 构造窗口并检查非空
4. 计算/保存统计量
5. DataLoader 构建
6. 模型、优化器、AMP（CUDA 时）初始化
7. epoch 训练 + dev 评估 + early stopping
8. 保存 `last.ckpt` 与 `best.ckpt`
9. 使用 best checkpoint 在 test 集评估并写出 `final_metrics.json`

日志文件：

- `metrics.jsonl`：每个 epoch 一行 JSON
- `resolved_config.yaml`：实际生效配置
- `final_metrics.json`：最终结果

## 8. 推理流程

推理主流程（`ocean_forecast/infer.py`）支持：

- 给定 `start_time` + `best.ckpt` 生成未来 `pred_len` 小时预测
- 可选 `--auto_shift_start`：当历史窗口缺失时向后寻找最近可用起点
- 输出 `.npz`（含 pred、forecast_hours、forecast_times、lat、lon、mask 等）
- 可选输出 `stats_csv`（每小时每变量海洋网格均值/最小/最大）

## 9. 配置文件

当前有两份主配置：

- `configs/convlstm_baseline.yaml`
  - `stride: 1`
  - `epochs: 20`
  - `device: auto`
- `configs/convlstm_mid_8gb.yaml`
  - `stride: 48`
  - `epochs: 3`
  - `device: cuda`

运行时可用 CLI 覆盖：

- 训练：`--output_dir --max_train_batches --max_eval_batches --epochs --patience`
- 推理：`--device --stats_path --auto_shift_start`

## 10. 现有产物与结果（outputs）

目录示例：

- `outputs/smoke_real`
- `outputs/smoke_wsl`
- `outputs/train_mid_8gb_full`
- `outputs/gpu_probe`

代表性结果（来自 `final_metrics.json`）：

- `outputs/smoke_real`：`test nrmse_mean ≈ 0.7960`（device=`cpu`）
- `outputs/smoke_wsl`：`test nrmse_mean ≈ 0.7960`（device=`cuda`）
- `outputs/train_mid_8gb_full`：`test nrmse_mean ≈ 0.5802`（device=`cuda`）

说明：`metrics.jsonl` 为追加写入，同一目录多次训练会叠加历史记录。

## 11. 测试现状

已执行：

```bash
pytest -q
```

结果：`4 passed in 2.56s`

覆盖点包括：

- zip NetCDF 解码、空文件跳过、填充值处理
- 连续窗口构造
- ConvLSTM 输出维度
- masked loss 与 nrmse 计算

## 12. 辅助脚本与非主链路文件

- `scripts/smoke_run.py`：
  - 自动执行 1 epoch 小训练（2 train batch + 2 eval batch）
  - 然后执行一次推理，适合快速验链路
- `build_numpy_data.py`：
  - 与主包解耦，面向另一类风浪/台风标注样本构建
  - 当前不被 `train.py` / `infer.py` 调用

## 13. 已识别的工程特征与注意事项

- 仓库当前有已修改但未提交文件：`README.md`
- `.gitignore` 忽略了 `*.zip`、`*.pyc`、`outputs`
- 主训练代码是模块化结构，便于替换模型与扩展数据读取器
- 当前缺少统一依赖清单文件，环境复现需手动对齐
- `configs`/默认配置里的 `root_dir` 中文字符串在部分终端显示为乱码，若出现路径找不到，应优先核对实际目录名是否为 `dataset/海域要素预测`

## 14. 推荐的文档外延（后续可补）

如需进一步完善为“交付级”文档，建议补充：

- 环境安装清单（含 CUDA/PyTorch 对应表）
- 数据字段字典（各变量单位、范围、缺测策略）
- 训练资源消耗基线（GPU 显存、耗时、吞吐）
- 结果对比表（不同 stride / hidden_dims / epochs）


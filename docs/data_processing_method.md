# 数据处理方法简介

本文档概述本项目的时空海洋数据处理流程，目标是将原始年度 NPY 数据转换为可直接训练/评估的样本序列。

## 1. 数据组织与读取

- 数据按年份存储为 `YYYY_data.npy`，形状为 `[T, 4, H, W]`，4 个基础通道为 `sst/sss/ssu/ssv`。
- 对应时间索引由 `YYYY_hours.npy` 提供，统一映射为小时级时间戳。
- 读取阶段会同时生成海域有效掩码（`mask`），仅在有效海域上计算统计量与损失。

## 2. 时间切分（Train/Dev/Test）

- 采用配置中的时间边界进行切分（`train_start/train_main_end/dev_start/dev_end/test_start/test_end`）。
- 切分结果是连续的帧引用序列，后续滑窗采样基于该序列执行。

## 3. 滑动窗口构建

- 每个样本由 `input_len + pred_len` 个连续小时帧组成。
- 窗口连续性要求：窗口内相邻帧时间差必须为 1 小时。
- 默认步长由 `data.stride` 控制；训练集可额外使用 `train_stride_schedule` 做分段稀疏/致密采样（按预测起点时间分段）。

## 4. 归一化与目标构建

- 在训练主集上统计 `input_mean/std` 与 `target_mean/std`（逐通道，海域有效格点）。
- 输入 `x` 与目标 `y` 均做标准化：

```text
x_norm = (x - input_mean) / input_std
y_norm = (y - target_mean) / target_std
```

- 目标支持两种模式：
  - `uv`：目标为 4 通道 `sst/sss/ssu/ssv`
  - `legacy_speed`：目标为 3 通道 `sst/sss/speed`，其中 `speed = sqrt(ssu^2 + ssv^2)`

## 5. 输入特征增强

在基础 4 通道之上，可按配置附加：

- 海域掩码通道（`add_mask`）
- 小时周期编码（`add_time_hour`，sin/cos）
- 年周期编码（`add_time_year`，sin/cos）

增强后输入通道数由：

```text
4 + [mask] + [hour_sin, hour_cos] + [year_sin, year_cos]
```

## 6. 训练阶段采样（可选）

- 可启用 `speed_rebalance` 按样本未来窗口速度统计进行分桶加权采样（`WeightedRandomSampler`）。
- 该机制只改变训练抽样分布，不改变 dev/test 评估流程与指标定义。

## 7. 产物与复用

预处理后会落盘以下工件，用于训练复现与快速重启：

- `split_refs.npz`：train/dev/test 索引
- `train_windows.npy`、`dev_windows.npy`、`test_windows.npy`：窗口起点
- `stats.npz`：归一化统计与掩码
- `manifest_exp.json`：配置摘要与窗口构建元信息（含分段 stride 信息）

该流程保证了数据切分、采样和归一化的一致性，便于进行可复现实验与模型横向对比。


# 评委：TOPIQ-NR（C 组回归型）

TOPIQ 的无参考（NR）分支：CFANet ＋ timm ResNet50，单张原分辨率整图一次前向给出一个质量标量。
本包复现 pyiqa 的 `topiq_nr`（KonIQ-10k 支），其余变体（`topiq_nr-flive` / `-spaq` / `-face` /
`topiq_fr` / `topiq_iaa`）未复现。与同属 C 组回归型的 MANIQA 相比输入口径相反：MANIQA 跑
20 个 224 crop 后加权平均，本评委只跑单张整图。

**污染状态：推定干净（中置信）。** 加载的是 pyiqa 自带权重，其训练配置的 `split_file` 用官方
split；中置信的理由是未走到逐图清单，无法确认那个 yml 就是生成这份 ckpt 的那一次跑。

## 复现对象与许可

Chen, Mo, Hou, Wu, Liao, Sun, Yan, Lin. *TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment*, IEEE TIP 2024（arXiv:2308.03060）。官方实现
[chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)。

| 对象 | 许可 | 说明 |
|---|---|---|
| 代码（`vendor_cfanet/`，拷自 pyiqa） | **PolyForm Noncommercial License 1.0.0** | 不可商用；再分发须随附许可全文与 Required Notice（Copyright (c) 2022 Chaofeng Chen） |
| 权重（pyiqa 权重仓） | **CC BY-NC-SA 4.0** | 不可商用；该仓为混合来源，不得据仓许可外推仓内每一份权重的条款 |
| 底座 ResNet50（timm / torchvision） | Apache-2.0 / BSD 3-Clause | 可商用 |

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | timm `resnet50`（`features_only`），全部中间尺度做跨尺度融合 |
| 权重文件 | `cfanet_nr_koniq_res50-9a73138b.pth`，**181,196,009 B** |
| SHA256 | `9a73138beb3748dd6a83365610ac88fa1cd53225342c77fc9ed44a3be461354c` |
| 来源 | pyiqa release 资产／HF `chaofengc/IQA-PyTorch-Weights`；两处长度字段均为 181,196,009，文件名后缀 `9a73138b` 即 SHA256 前 8 位，与 HF 的 `X-Linked-Etag` 逐位相同。本机无法直连 github.com / huggingface.co，下载经 `hf-mirror.com` |
| 落地位置 | `D:\models\TOPIQ-NR\`（**不随本包释出**） |
| 加载 | `strict=True`，`weight_keys="params"`；`BACKBONE_PRETRAIN = False`（ckpt 含 `semantic_model.*`，timm 的 ImageNet 权重会被整体覆盖） |
| 输入与输出 | `num_crop=1`、`test_img_size=None`、`crop_size=256`（仅记录）；`num_class=1`，`dist_to_mos` 退化为恒等 |

`--inspect-ckpt` 实测：参数键 **524** 个，其中 `semantic_model.*` **318** 个，前 3 键 `h_emb` /
`w_emb` / `semantic_model.conv1.weight`；架构键集与 ckpt 键集一致（`True`）⇒ `strict=True`
加载通过即 vendored 架构与上游逐键等价的证据。

## 预处理与打分口径

输入为**原分辨率整图**，不缩放、不裁剪；归一化 `(x − ImageNet_mean) / ImageNet_std` 在模型
`preprocess` 内部完成。打分取模型最后一层标量，名义区间 **≈ [0, 1]**（pyiqa 写作
`score_range: '~0, ~1'`），量纲 **0–1**；KonIQ 的 0–100 MOS 只在算指标时使用，不进入推理。
`--list` 用于取官方子集：KonIQ-10k 目录含全部 **10,073** 张，对目录 glob 会把训练集混进 test
口径；取 `set == test` 时必须传名单。

## 运行栈

本节的 L4 数字与耗时为**本机栈**：Python 3.11.15 / torch **2.6.0+cu124** / GTX 1650 4 GB，`--dtype` 默认 **fp32**。环境适配：`infer.py` 把 stdout/stderr 改为 UTF-8——Windows 控制台默认 GBK，装不下 `⇒`、`≤` 等符号，不改会 `UnicodeEncodeError`；纯输出路径，不改变数值。

## 本包包含什么

本目录随包只有 `infer.py`（图 → 分数 ＋ 标量）与本 README。**不随包释出**：`vendor_cfanet/`、
下载脚本、运行脚本与输出缓存。`vendor_cfanet/` 是导入依赖，取自 `chaofengc/IQA-PyTorch`
commit `6e48c5ca9e3e58bd632af52f06a2ca976ccf2722` 的 `pyiqa/archs/`；未 vendor 的 Swin / CLIP /
gfiqa 人脸支改为同签名占位函数，一旦被走到即 `NotImplementedError`，不会静默走错路。

| 文件 | 上游路径 | 改动／正文摘要 |
|---|---|---|
| `cfanet_arch.py` | `topiq_arch.py` | 导入拉平、摘注册表装饰器、`DEFAULT_CACHE_DIR` 本地化、CLIP/Swin/facexlib 三处导入改同签名占位 |
| `arch_util.py` | `arch_util.py` | 去头注、行尾归一后正文 10,372 B / sha256 `6764807c…58f0393` |
| `constants.py` | `constants.py` | 正文 636 B / sha256 `811a450a…e55b5f16681`，等于上游 blob |

用法：`python infer.py --image <图或目录> --out <缓存>.jsonl`；结构核对用
`python infer.py --inspect-ckpt`（只打印 ckpt 键集，不跑图）。

## 实测

**口径**：KonIQ-10k 官方 test **2,015 张**（512×384，按 `koniq10k_distributions_sets.csv` 的
`set` 列），MOS 取同一 CSV 的 `MOS` 列（0–100）。TID2013 可用集 **2,687 / 3,000 张**（384×512
原生），排除内容 i01/i25 全部 **240** 张 ∪ `_BROKEN_FILES.txt` 其余 **73** 张；TID2013 无官方
split，`set` 一律记 `test`（对可用集整体评估），MOS 取 `mos_with_names.txt`（0–9 制，实测
0.24242 .. 7.21429，无缺失）。两者分别 2,015/2,015、2,687/2,687 全部命中，退出码 0。
标签件 `tid2013_eval_labels.csv` 的 sha256 为
`5a765bfe1bcaf63e773718714602c4c4886357332b97014934e23fe5e34af8d3`，名单件（2,687 行）为
`cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`；两者均不进本包。

| 指标 | KonIQ（2,015 张，512×384 native） | TID2013（2,687 张，384×512 native） |
|---|---|---|
| SRCC | **0.9298** | **0.4719** |
| KROCC | 0.7726 | **0.3349** |
| PLCC（raw） | **0.9437** | **0.5808** |
| PLCC（4 参数 logistic） | 0.9440 | **0.6226** |

预测范围：KonIQ **0.1023 .. 0.8590**；TID2013 **0.0527 .. 0.8031**（0–1 制）。以上均出自本机栈。

## 与 pyiqa 基准的对照

pyiqa 自带 `tests/NR_benchmark_results.csv`，`topiq_nr` 行在 KonIQ 列为 PLCC **0.9436** /
SRCC **0.9299** / KRCC **0.7727**；本机 0.9437 / 0.9298 / 0.7726，差 **+0.0001 / −0.0001 /
−0.0001** ⇒ **一致到 1e-4**。**该一致性只在本机栈成立**（Python 3.11.15 / torch 2.6.0+cu124 /
GTX 1650 4GB）；换栈有实测约 5e-3 量级的位移，故不得无条件写「复现了 pyiqa 基准」。

同一张 CSV 的 `tid2013` 列为 PLCC **0.5625** / SRCC **0.4452** / KRCC **0.3143**，本机
0.5808 / 0.4719 / 0.3349，残差 **+0.0183 / +0.0267 / +0.0206**，三项同号且为正。这只是旁证：
该列覆盖全部 3,000 张（不排除 i01/i25 与损坏文件），我们只跑 2,687 张，且其输入口径只与
本机自述相符、未逐项实测。**这是同一份文件的两列，不是同一次测量。**

论文 TABLE VIII 的 KonIQ-10k official-split 自报值为 TOPIQ (ResNet50) PLCC **0.941** /
SRCC **0.928**，本机差 **+0.0027 / +0.0018**，量级 1e-3；表格自报值与 pyiqa CSV 并非同一次
测量，能对上的那一份必须明说是哪一份。

## 已知限制

1. **集合不同**：pyiqa 的 TID2013 meta_info 按 `mos_with_names.txt` 逐行生成，覆盖全部 3,000 张；我们只跑 2,687 张。其评测时是否再取子集，未核。
2. **输入口径未逐项比对**：pyiqa 自述 *"Full image single input"*、*"PLCC without any correction"*，与本机 `num_crop=1` 一致，但只是自述相符。
3. **i01/i25 的选择性偏差未检验**——TID2013 的数字不得默认与其余 23 个内容同分布。
4. **两列不可直接比**：数据集、标注谱系（TID2013 是成对比较协议）、失真类型与 MOS 量纲均不同；并列只为留跨域迁移锚点，不构成「下降 = 复现变差」的判断。
5. **污染定性为中置信**：未走到逐图清单，不能把「推定干净」升格为确定干净。

## 冒烟证据

六张冒烟图（GTX 1650）：`p1_orig.png` 0.6576、`photo1.jpg` **0.6576**、`p1_blur1.png` 0.3473、
`p1_blur2.png` 0.1929、`p1_jpeg8.png` 0.3906、`p1_noise.png` 0.2452。质量梯度正确：原图 0.658 >
轻模糊 0.347 > 重模糊 0.193，原图 > JPEG8 0.391 > 噪声 0.245。
`p1_orig.png` 与 `photo1.jpg` 像素完全相同、仅编码不同，分数逐位相同 ⇒ 管线确定、不受 JPEG/PNG
解码路径影响。六张图只说明值得做，不构成证据。

## 耗时

单图均值（预热后，2011 张）**0.0523 s**，中位 / 最大 / 最小 0.050 / 0.180 / 0.030 s；2015 张合计
**≈ 107 s**（含加载 1.0 s）。同为 C 组回归型的 MANIQA 单图均值 3.891 s ⇒ 约 **74 倍**差距：
MANIQA 跑 20 个 ViT-B/8 crop，本评委只跑 1 张原分辨率图的 ResNet50，同机制不等于同吞吐。
TID2013 侧加载 0.9 s，单图中位 0.0500 s/图、均 0.0460 s/图，2,687 张墙钟 2.3 min。

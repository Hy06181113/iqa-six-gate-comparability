# 评委：HyperIQA（C 组，回归型）

**C 组（回归型）**，九个本地评委之一。它用 ResNet50 骨架 ＋ 自适配超网络（HyperNet）做无参考回归，输出 0–1 分数；选它是因为它补上 C 组的 HyperNet 分支，且与 MANIQA / MUSIQ 同组而骨干、训练来源不同。**污染状态：推定干净（中置信）**。

**污染定性必须看我们实际加载的那份权重，不能看原论文。** 我们加载的不是原论文的权重：原作者从未发布 KonIQ-10k 权重，用的是 **pyiqa 自己重训**的那一份。pyiqa `docs/benchmark.rst` 明文把 `hyperiqa` 列在「不提供官方权重、由我们重训」一栏（同栏还有 `cnniqa` / `dbcnn`），其训练配置 `options/train/HyperNet/train_HyperNet.yml` 的 `split_file` 用官方 split（`./datasets/meta_info/koniq10k_official.pkl`），因此推定它未见过我们的 test 图。中置信的理由：未走到逐图清单，无法确认该 yml 就是生成这份 ckpt 的那一次跑（`004_HyperNet_KonIQ10k` 与权重文件名之间没有可核对的链接）。

## 复现对象与许可

Su, Yan, Zhu, Zhang, Ge, Sun, Zhang. *Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network*, CVPR 2020, pp. 3667–3676. 官方实现：<https://github.com/SSL92/hyperIQA>。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者侧代码（`SSL92/hyperIQA`） | **MIT** | 可商用；但本封装实际复制的是 pyiqa 的代码 |
| 本封装实际复制的代码（`vendor_hypernet/`，pyiqa） | **PolyForm Noncommercial License 1.0.0** | **不可商用**；再分发须随附许可全文与 Required Notice |
| 权重（pyiqa 权重仓 `chaofengc/IQA-PyTorch-Weights`） | **CC BY-NC-SA 4.0** | **不可商用**；仓内混合来源，不得据该仓许可外推每一份权重 |
| 底座模型（ResNet50，torchvision 预训练） | **BSD 3-Clause** | 可商用 |

本封装实际加载的是 pyiqa 的代码，许可按 pyiqa 那一行判；**不得**据作者仓库的 MIT 外推。

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | ResNet50（`timm.create_model(..., features_only=True)`） |
| 权重文件 | `HyperIQA-resnet50-koniq10k-c96c41b1.pth`，**109,834,491 B** |
| SHA256 | `c96c41b1094c8c1ed86cbd661941717f95e24e09e791b2c512d4ffbbde7253ed` |
| 来源 | `https://hf-mirror.com/chaofengc/IQA-PyTorch-Weights/resolve/main/HyperIQA-resnet50-koniq10k-c96c41b1.pth`（经镜像；上游 release 为同一文件） |
| 落地位置 | `D:\models\HyperIQA\`（**不随本包释出**） |

下载走 `hf-mirror.com`；`X-Linked-Size` = **109,834,491 B**、`X-Linked-Etag` = `c96c41b1…253ed`，与文件名后缀 `c96c41b1`（SHA256 前 8 位）逐位相同。**特殊加载项**：`BACKBONE_PRETRAIN = False`——上游硬编码 `pretrained=True`；本环境 `huggingface.co` 不可达、本机 HF 缓存无该权重，且 ckpt 自带全部 `base_model.*` 参数（**318 / 355** 键），`strict=True` 整体覆盖。

`--inspect-ckpt` 实测：顶层 `dict`、外层键 `params`、参数键 **355**、`base_model.*` **318**、`架构键集 == ckpt 键集: True` ⇒ 骨架权重在 ckpt 内，且 vendored 架构与上游逐键一致。这是本目录唯一一处与上游不同的、可能影响行为的改动。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 模型配置 | `type: HyperNet`、`base_model_name: resnet50`、`num_crop: 25`（5×5 均匀网格）、输出 `score_range: '~0, ~1'` |
| 单 crop 尺寸 | **224×224**（`uniform_crop(..., 224, self.num_crop)`，硬编码） |
| 输入 | 原分辨率整图（KonIQ 512×384 / TID2013 384×512），**不缩放**；25 个 crop 的分数取均值 |
| 归一化 | ImageNet mean/std，在模型 `preprocess` 内部完成 |
| 取官方 test 子集 | KonIQ-10k 目录含全部 10,073 张，**必须用 `--list` 限定 `set == test`**；对目录 glob 会把训练集混进 test 口径 |

## 运行栈

| 项 | 值 |
|---|---|
| 主实测栈（本机） | Python **3.11.15** / torch **2.6.0+cu124** / GTX 1650 4 GB，**fp32**；KonIQ 与 TID2013 两节数字均出自这一段栈 |
| pyiqa 自身代码对照栈（租卡） | torch **2.0.1+cu117** / RTX 3090（Python / dtype 未随此次记录）；仅用于「与 pyiqa 基准的对照」中的那组重跑 |

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → 分数）与本 README。**不随包释出**：

- `vendor_hypernet/`——三份文件逐字复制自 pyiqa commit `6e48c5ca9e3e58bd632af52f06a2ca976ccf2722`（2026-08-30，取回通道 `cdn.jsdelivr.net`）；**不 vendor 就不是复现**：
  - `hypernet_arch.py` ← `pyiqa/archs/hypernet_arch.py`（改动：导入拉平、去注册表装饰器、`pretrained=True` → `BACKBONE_PRETRAIN`；模型定义其余一字未改。正文 sha256 `2c149165…599b3f`，正文 8,394 B / 上游 blob 8,210 B，文件总 10,035 B）。
  - `arch_util.py` ← `pyiqa/archs/arch_util.py`（继承 maniqa 副本的两处补丁：删远程下载导入、URL 分支改抛错；正文 10,372 B，sha256 `6764807c…58f0393`）。
  - `constants.py` ← `pyiqa/archs/constants.py`（无改动；636 B，sha256 `811a450a…5f16681`，等于上游 blob）。
- `download_weights.py`（多线程 Range ＋ 断点续传 ＋ 签名过期自动刷新 ＋ 大小/摘要校验）与权重文件本身。
- `run_local.sh`、`runs/` 缓存。

正文 sha256 的口径：去掉文件头连续 `#` 注释块（止于第 2 个 `# ---` 行）后计算；「总大小 / 正文 / 上游 blob」是三个不同的数。

## 实测：KonIQ-10k 与 TID2013

**口径**：KonIQ-10k 官方 test **2,015 张**（512×384 native，MOS 0–100，按 `koniq10k_distributions_sets.csv` 的 `set` 列）；TID2013 可用集 **2,687 / 3,000 张**（384×512 native；排除 i01/i25 全部 **240** 张 ∪ `_BROKEN_FILES.txt` 其余 **73** 张；TID2013 无官方 split，`set` 一律 `test`）。两段均为**本机栈、fp32**，预处理均为原分辨率整图、不缩放。TID2013 MOS 取 `mos_with_names.txt`（0–9 制，实测 0.24242 .. 7.21429，无缺失）；名单由排除清单算出，不是对目录 `*.bmp` glob。标签件 sha256 `5a765bfe1bcaf63e773718714602c4c4886357332b97014934e23fe5e34af8d3`；名单件（2,687 行）sha256 `cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`。

| 指标 | KonIQ（2,015 张，512×384 native） | TID2013（2,687 张，384×512 native） |
|---|---|---|
| SRCC | **0.9019** | **0.4721** |
| KROCC | **0.7314** | **0.3323** |
| PLCC (raw) | **0.9218** | **0.5744** |
| PLCC (4 参数 logistic) | 0.9219 | **0.6097** |

预测范围 KonIQ **0.1590 .. 0.8412**（官方标签 4.1674 .. 88.2438，0–100 制）；TID2013 **0.1211 .. 0.7845**（0–1 制）。TID2013 加载 **0.8 s**；单图中位 **0.1300 s/图**、均 0.1331 s/图（最慢 3.06 s）；墙钟 **6.6 min**，2,687/2,687 全部命中、退出码 0。

## 与 pyiqa 基准的对照

同一张 pyiqa 基准表 `tests/NR_benchmark_results.csv` 的 `hyperiqa` 行（KonIQ 列）：PLCC **0.9233** / SRCC **0.9040** / KROCC **0.7336**。该表的协议（`docs/benchmark.rst`）为 NR 测官方 split、PLCC 不做校正、full image single input。本机对应 0.9218 / 0.9019 / 0.7314 ⇒ 残差 **−0.0015 / −0.0021 / −0.0022**（约 **2e-3**）。

| 来源 | PLCC | SRCC | KROCC | 与本机的差（PLCC / SRCC / KROCC） |
|---|---|---|---|---|
| pyiqa 基准 CSV | 0.9233 | 0.9040 | 0.7336 | −1.5e-3 / −2.1e-3 / −2.2e-3 |
| **pyiqa 自身代码**（租卡栈，2,015 张） | **0.9227** | **0.9030** | **0.7331** | −0.9e-3 / −1.1e-3 / −1.7e-3 |
| 本封装（本机栈） | 0.9218 | 0.9019 | 0.7314 | — |

pyiqa **0.1.16** 自身代码落在本封装与 CSV 之间（PLCC 离 CSV **6e-4**、离本封装 **9e-4**），两者同量级、分不开。同一次核验还量出：同一组图、图像 sha256 逐字节相同、协议字段逐项相同，**仅换栈**（本机 torch 2.6.0+cu124/1650 ↔ 租卡 2.0.1+cu117/3090），MUSIQ 的 SRCC 就位移 **≈5e-3**。⇒ 本评委与 CSV 的残差 ≈2e-3（与 pyiqa 自身代码 ≈1e-3）**小于实测栈差（≈5e-3）**，被栈差完全淹没，**不再具备区分「实现差异」与「换机差异」的能力**。因此只能写「**同量级**」，**不得**写「复现了 pyiqa 的基准」，也不得据此判断实现层面（含负号方向）的任何结论。

对照对象与量级：MUSIQ 0.8958 vs 0.8958（**0.0000**，四位小数全同）、TOPIQ-NR 0.9437 vs 0.9436（**+0.0001**，≤1e-4、舍入级）、**HyperIQA 0.9218 vs 0.9233（−0.0015，~2e-3，同量级）**、MANIQA 0.9291 vs 0.9133（+0.0158，~1e-2）。前两者可写「一致到 1e-4」，本评委只能写「同量级」。

**残差从何而来尚未查明。** 已排除 split（MUSIQ 四位小数全同，把「我们的 2,015 张 == pyiqa 的 test 集」钉死）与输入协议：pyiqa 的 `koniq10k` 配置为 `./datasets/koniq10k/512x384` + `official_split` + `phase: test`，无 `test_img_size`、无 resize/crop 覆盖，数据集侧无任何变换，模型侧 `hyperiqa` 无额外覆盖 ⇒ 喂的是 512×384 整图，与本封装 `infer.py` 一致；批大小也不是来源（PLCC 是逐图分数的确定函数）。剩下的可能只有「基准 CSV 出自另一版本 / 另一次跑」。

TID2013 侧的 pyiqa 基准 CSV（同一张表的 `tid2013` 列，用的也是同一份 pyiqa 重训权重，KonIQ 列亦然）为 PLCC **0.5627** / SRCC **0.4537** / KRCC **0.3177**；本机 0.5744 / 0.4721 / 0.3323 ⇒ 残差 **+0.0117 / +0.0184 / +0.0146**（三项同号）。方向与「我们的集合是它的子集、且剔除了 i01/i25 与损坏文件」相符，但该解释未验证，**只能当旁证，不作判据**，也不作「复现成功」的对外宣称。

## 已知限制

1. **污染判定是中置信的「推定干净」**：pyiqa 只给了训练配置与 pkl 名，未给逐图清单；无法确认 `train_HyperNet.yml` 就是生成这份 ckpt 的那一次跑。不得写成「确认干净」。
2. **KonIQ 残差已被栈差淹没**：≈2e-3 < ≈5e-3，不能再用于区分实现差异与换机差异，也不能用负号推断「权重未见过 test 图」的方向。
3. **TID2013 集合不同**：pyiqa 的 meta_info 按 `mos_with_names.txt` 逐行生成，覆盖全部 **3,000 张**（不排除 i01/i25、不排除损坏文件）；我们只跑 **2,687** 张。其评测时是否再按 `tid2013_seed123.pkl` 取子集，未核。
4. **TID2013 输入口径未逐项比对**：本封装走 `num_crop=25`；pyiqa 自述「multi-patch testing only when it is necessary」，HyperIQA 在它那儿取几个 crop 未核。
5. **i01/i25 的选择性偏差未检验**：TID2013 的数字不得默认与其余 23 个内容同分布。
6. **两列不可直接比**：数据集、标注谱系（TID2013 是成对比较协议）、失真类型、MOS 量纲均不同。并列只为给跨域迁移留锚点，不构成「下降 = 复现变差」的判断。

## 其它证据

### 冒烟：六张图上的对照

| 图 | HyperIQA | MUSIQ | TOPIQ-NR | MANIQA | CLIP-IQA | Q-ReAlign Mini |
|---|---|---|---|---|---|---|
| `p1_orig.png` | **0.5591** | 72.4179 | 0.6576 | 0.5717 | 0.9630 | 0.9040 |
| `photo1.jpg` | **0.5591** | 72.4179 | 0.6576 | 0.5717 | 0.9630 | 0.9040 |
| `p1_blur1.png` | 0.4180 | 23.6872 | 0.3473 | 0.3331 | 0.8480 | 0.3690 |
| `p1_blur2.png` | 0.4114 | 17.3040 | 0.1929 | 0.3540 | 0.3024 | 0.0878 |
| `p1_jpeg8.png` | 0.4031 | 32.8718 | 0.3906 | 0.4124 | 0.2069 | 0.2613 |
| `p1_noise.png` | 0.2248 | 23.5072 | 0.2452 | 0.2359 | 0.6960 | 0.2357 |

`p1_orig.png` 与 `photo1.jpg` 像素完全相同、仅编码不同，HyperIQA 给出**逐位相同**的 0.5591 ⇒ 管线确定性、且不受 JPEG/PNG 解码路径影响。质量梯度正确（原图 > 各种失真）。一处值得记的观察：HyperIQA 把最重的模糊 `blur2`（0.4114）排在 `jpeg8`（0.4031）之上，`blur1` 与 `blur2` 几乎并列（0.4180 / 0.4114）。六张图只说明值得做，不构成任何结论。

### 耗时（GTX 1650，512×384 native，fp32）

| | HyperIQA |
|---|---|
| 单图均值（预热后） | **0.1300 s** |
| 中位 / 最大 / 最小 | 0.130 / 0.170 / 0.110 s |
| 2,015 张合计 | **≈262 s（实测）** |
| 26,180 张（全量口径） | **0.9 h**（按相同速率外推） |

六个已实现评委的速率跨度是 **86 倍**（0.0450 → 3.891 s/图）；最快的三个（MUSIQ / TOPIQ-NR / HyperIQA）与最慢的 MANIQA 同属 C 组 ⇒「组」不能外推耗时。根因是输入口径不同：HyperIQA 跑 **25 个 ResNet50 crop 再过超网络**。

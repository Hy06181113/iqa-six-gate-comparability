# 评委：MANIQA（C 组回归型，KonIQ-10k 支）

MANIQA 属于 **C 组（回归型）**，输出单个标量，名义区间约 [0,1]；本封装复现作者在 KonIQ-10k
上微调的 `ckpt_koniq10k.pt` 一支，PIPAL / KADID 两支未复现。它是本地评委里第一个在 KonIQ-10k
上训练过的。**污染状态：已污染。** 该 checkpoint 的训练 split 是全库随机 80/20（`seed=20`，未遵守
官方 split），实测覆盖我们官方 test 的 **1,607 / 2,015 张（79.8%）**，只有 **408 张**是它没见过的图。

## 复现对象与许可

Yang, Wu, Shi, Lao, Gong, Cao, Wang, Yang. *MANIQA: Multi-dimension Attention Network for
No-Reference Image Quality Assessment*, CVPR Workshop 2022（**NTIRE2022 NRIQA 挑战赛冠军**）。
官方实现：<https://github.com/IIGROUP/MANIQA>。本封装走 pyiqa 的 PyTorch 实现，架构代码逐字取自 pyiqa。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者仓库 `IIGROUP/MANIQA` 代码 | **Apache-2.0** | 本封装未直接使用作者代码，仅作方法出处 |
| 本封装实际复制的 pyiqa 代码 | **PolyForm Noncommercial License 1.0.0**（Required Notice: Copyright (c) 2022 Chaofeng Chen） | **不可商用**；再分发须随附许可全文与 Required Notice |
| 权重 `ckpt_koniq10k.pt` | pyiqa 权重仓 **`cc-by-nc-sa-4.0`** | 混合来源仓，仓内 README 声明部分权重由他人训练、须尊重原许可；**不可商用** |
| 底座模型 ViT-B/8（timm） | **Apache-2.0** | 可商用 |

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | `vit_base_patch8_224`（ViT-B/8，取第 6–9 个 Block 输出拼接） |
| 权重文件 / 字节数 | `ckpt_koniq10k.pt`，**543,335,435 B**（作者 GitHub release 的 `Content-Length` 与 HF 镜像的 `X-Linked-Size` 相同） |
| SHA256 | **作者未公布**，无先验摘要可校验；权重下载脚本（不随包）落盘后会打印实测值作为本机基线 |
| 来源 / 落地 | 作者 release `IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt`；本机经 `hf-mirror.com` 从 `chaofengc/IQA-PyTorch-Weights` 取同一文件，落地 `D:\models\MANIQA\ckpt_koniq10k.pt`（**不随本包释出**） |
| 加载方式 | `strict=True` 加载裸 state_dict；跳过 timm 的 ImageNet 初始化（`pretrained=False`） |

完整性：本模型 **135.7 M** 参数 × fp32（4 B）= **542.9 MB**，与 ckpt 的 **543.3 MB**
（543,335,435 B）基本相等 ⇒ ckpt 含**全部**参数（含 `vit.*`），`strict=True` 会整体覆盖 timm
初始化，跳过下载不影响数值；若换成不含完整键集的 ckpt，`strict=True` 会直接报错，不会静默出错。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 输入 / 缩放 | PIL `convert("RGB")` → `np.float32 / 255` → CHW，[0,1]；**不缩放** |
| 裁剪 | `uniform_crop(224, crop_num=20)` 等步长网格，**无随机性** |
| 归一化 | `(x − 0.5) / 0.5`，在模型 `forward` **内部**完成且只做一次；外部不做任何归一化 |
| 聚合 | 20×784 个 patch **一次联合加权平均** → 一个标量 |
| `scale` | **0.8**，硬编码、无 CLI 开关；类默认值 0.13 不会进 `state_dict`，传错会静默算错 |
| 输出 | 单个标量，名义区间 ≈ [0,1] |

## 运行栈

**本机栈**（KonIQ / TID2013 L4、冒烟、耗时）：Python 3.11.15 / torch **2.6.0+cu124** /
GTX 1650 4 GB / fp32。**租卡栈**（pyiqa **0.1.16**）只用于「pyiqa 自身代码」一行，**跨栈数字不混表**。

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → 分数标量）与本 README；`vendor_iqa/`、权重下载
脚本、官方协议复刻脚本、本机运行脚本、缓存目录与权重文件均**不随包释出**。

`vendor_iqa/`（不随包）取自 `chaofengc/IQA-PyTorch` 的 commit
`6e48c5ca9e3e58bd632af52f06a2ca976ccf2722`（2026-08-30），对应上游
`pyiqa/archs/{maniqa_arch,arch_util,constants,maniqa_swin}.py`。补丁：导入拉平、去掉
`ARCH_REGISTRY` 装饰器、`pretrained` 改为参数、删除下载工具导入、`load_pretrained_network`
的 URL 分支改抛 `NotImplementedError`；其余逐字节相同。`infer.py` 的
`from vendor_iqa import MANIQA` 需要该目录存在。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl --device cuda
```
## 实测

**口径**：KonIQ-10k 官方 test **2,015 张**（512×384）；TID2013 **可用集 2,687 / 3,000 张**
（384×512 native，无官方 split，`set` 记 `test`）。全部为**本机栈**，预处理 `scale=0.8` +
`uniform_crop` 20 + fp32 两次 run 一致。

TID2013 的标签件 `tid2013_eval_labels.csv` sha256 为
`5a765bfe1bcaf63e773718714602c4c4886357332b97014934e23fe5e34af8d3`，名单件（2,687 行）为
`cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`；两者均不进本包。

| 指标 | KonIQ-10k 官方 test（2,015 张） | TID2013 可用集（2,687 张） |
|---|---|---|
| SRCC | **0.9042** | **0.4862** |
| KROCC | 0.7347 | **0.3421** |
| PLCC（raw） | **0.9291** | **0.5542** |
| PLCC（4 参数 logistic） | **0.9349** | **0.5887** |

KonIQ 这一列并非泛化指标：**1,607 / 2,015（79.8%）是该 checkpoint 的训练图**；干净子集
**408 张**上为 SRCC **0.8960** / PLCC(raw) **0.9246** / PLCC(log) **0.9283**。污染使 SRCC 虚高
约 0.008、PLCC(log) 虚高约 0.006；「封装实现无误」不受影响，但报数必须标口径。

TID2013 侧：2,687 / 2,687 张全部命中、退出码 0；加载 **2.7 s**，单图稳态中位 **3.89 s/图**，
墙钟 **2 h 54 min 59 s**，预测范围 **0.0501 .. 0.6806**。作者自报 SRCC **0.930** / PLCC **0.946**，
与本机差 **−0.0258 / −0.0169**（PLCC raw）与 **−0.0111**（PLCC log）；口径不同，不能当门槛。

## 与 pyiqa 基准的对照

同一批 KonIQ 官方 test 2,015 张、512×384：

| 来源 | 栈 | PLCC | SRCC | KROCC |
|---|---|---|---|---|
| pyiqa 基准 CSV | — | 0.9133 | 0.8934 | 0.7198 |
| **pyiqa 自身代码**（同批 2,015 张） | 租卡栈 | **0.9271** | **0.9015** | **0.7310** |
| 本课题封装 | 本机栈 | 0.9291 | 0.9042 | 0.7347 |

pyiqa 自己的代码给出的是**本课题的数**（差 **2.0e-3 / 2.7e-3 / 3.7e-3**），而不是它自己 CSV
的数（差 **1.4e-2 / 8.1e-3 / 1.1e-2**）。按预注册规则判定：**CSV 那一行过时，本课题实现无
差异**；残留的 2–4e-3 混着两栈之差。**仍不得写「复现了 pyiqa 基准 CSV」——pyiqa 自己的代码
也复现不出它。**

TID2013 侧，pyiqa CSV 的 `maniqa` 行为 PLCC **0.457** / SRCC **0.4515** / KRCC **0.318**，
本机为 0.5542 / 0.4862 / 0.3421，残差 **+0.0972 / +0.0347 / +0.0241**；该行同样已判为过时，
且 pyiqa 的清单覆盖全部 **3,000 张**（不排除 i01/i25 与损坏文件），我们只跑 **2,687 张**。

## 已知限制
1. **污染口径**：报 KonIQ 的 0.9042 时必须同时说明「其中 79.8% 为该 checkpoint 的训练图」，
   不得当作泛化指标；干净子集 408 张的 0.8960 与作者的 2,015 张不是同一集合，不宜直接对减。
2. **TID2013 无官方 split**：可用集 2,687 / 3,000，排除内容 i01/i25 全部 **240 张**与
   `_BROKEN_FILES.txt` 其余 **73 张**；i01/i25 的选择性偏差**未检验**。
3. **两列不可直接比**：数据集、标注谱系（TID2013 是**成对比较**协议）、失真类型、MOS 量纲
   均不同，且污染状态相反；TID2013 那列是干净跨库数字，但性质是**分数缓存 + 复现证据**，不是泛化结论。
4. 官方 README 的 I01 系列数值用本封装（协议 A）与官方流程（协议 C）都对不上：0.8141 给了
   失真图 I01_01_02，而无失真参考图 I01 只跑出 0.5763。推测是旧权重时代的数，**未验证**，不能用作基准。
5. `kunkun.png` 的 0.4% 硬门槛验证的是封装管线（架构 + 权重加载 + `scale` + 归一化路径），
   **不是精度**；该图很可能也在训练集里。冒烟只有 6 张图，不足以支撑结论。

## 复现验证：官方 demo 默认输入（硬门槛）

第一层基线的硬门槛是「对官方 demo 默认输入的复现精度」。官方 `predict_one_image.py` 的
默认图是 `test_images/kunkun.png`、默认权重是 `./ckpt_koniq10k.pt`，与本封装同图、同权重。

| 协议 | crop | 聚合 | kunkun.png | 与官方 README 之差 |
|---|---|---|---|---|
| 官方 README | — | — | **0.3398** | — |
| **C：官方 demo 逐字复刻** | 20 个随机（seed 20） | 逐 crop 平均再取均值 | **0.3384** | **−0.0014（0.4%）** |
| **A：本封装（pyiqa `uniform_crop`）** | 20 个等步长网格 | 20×784 patch 联合加权平均 | 0.3427 | +0.0029（0.9%） |

**分辨率口径**：对比对象是官方 demo 的默认输入（原图 + 224×224 crop，不做整图 resize），
不是 512×384 的全量评测口径。协议 C 用官方代码的原始流程（`setup_seed(20)`、20 次随机 crop、
逐 crop forward 后取均值），在官方示例图上命中 0.4% ⇒ 架构、权重加载、`scale=0.8`、归一化路径
全部确认无误；本封装选协议 A，网格确定性，A 与 C 在各图上差 ≤ 0.01。

## 冒烟与耗时

协议 A（`uniform_crop` 20，`scale=0.8`，fp32），本机 GTX 1650；图集 6 张 512×512。

| 图 | 分数 | 期望梯度 | 实测名次 |
|---|---|---|---|
| `p1_orig.png` | 0.5717 | 最高 | 1（并列） |
| `photo1.jpg` | 0.5717 | 最高 | 1（并列） |
| `p1_jpeg8.png` | 0.4124 | 低 | 2 |
| `p1_blur2.png` | 0.3540 | 最低 | 3 |
| `p1_blur1.png` | 0.3331 | 中低 | 4 |
| `p1_noise.png` | 0.2359 | 中 | 5 |

`p1_orig.png` 与 `photo1.jpg` 像素完全相同，分数逐位相同（0.571681），说明封装是确定性的。
MANIQA 在这 6 张上**没有**给出期望的质量梯度：JPEG 排第二、两档模糊次序颠倒、噪声垫底。
耗时：冒烟加载 **2.2 s**、约 **3.8 s/图**，显存峰值 **2.85 / 4.29 GB**（fp32 够用）；
TID2013 全量 2,687 张的墙钟与吞吐见上「实测」。

---
arXiv分类: "cs.CV"
领域: "3D 视觉 / 点云场景理解"
发表时间: "2026-04-21 (v1)；2026-08-25 (v2)"
读文章时间: "2026-08-27"
标题: "Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding"
链接: "https://arxiv.org/abs/2604.19609"
Zotero: "未入库（本机 Zotero 9.0.6 不支持 Local API 写入，需 Zotero 10+）"
页码说明: "PDF 物理页码"
---

# Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding

## 1. TL;DR

这篇文章提出 **Volume Transformer (Volt)**：把 2D 视觉里最朴素的 ViT 式 vanilla Transformer 编码器直接搬到 3D 场景理解上——把点云切成「体素块 token」、用全全局自注意力 + 3D 版 RoPE，取代 3D 领域一直依赖的卷积/局部注意力 backbone。核心贡献不只是架构简单，还有一套**数据高效训练配方**（强增强 + 正则化 + 卷积 teacher 蒸馏），并证明了这类"少归纳偏置"的架构在**多数据集联合训练**下比领域专用 backbone 更吃数据红利，最终在室内外语义/实例分割多个基准上刷新 SOTA（ScanNet 80.5、ScanNet200 41.6、nuScenes 82.2 mIoU），同时推理更快、内存更省。

## 2. 作者与单位

- Kadir Yilmaz *（RWTH Aachen University；第一作者共同贡献）
- Adrian Kruse *（RWTH Aachen University；第一作者共同贡献）
- Tristan Höfer（RWTH Aachen University）
- Daan de Geus（Eindhoven University of Technology）
- Bastian Leibe（RWTH Aachen University）
- 项目页：vision.rwth-aachen.de/Volt（P.1）

## 3. 紧密相关的工作与交叉点

这篇文章站在三条线的交叉点上：

1. **ViT 的"少归纳偏置 + 大数据"范式** [2]：ViT 在数据少时打不过 CNN，但数据/算力上去后反超。作者想验证 3D 里同样成立。
2. **3D backbone 的演化**：从 PointNet/PointNet++（集合处理）→ 连续卷积 → **稀疏卷积 U-Net**（MinkUNet 等）→ 局部注意力混合架构（Stratified Transformer、Swin3D、OctFormer、Point Transformer v3 = 当前 SOTA，用空间填充曲线做局部注意力）。共同点是都保留 U-Net 分层 + 局部算子 + 强领域先验。
3. **2D 的数据高效训练配方** [33, 39]（DeiT 蒸馏、强增强、正则化）和 **RoPE 位置编码**（LLM [25,26] 和 ViT [27] 的标准选择）。

与最近工作的差异：LitePT [24]（ETH 的轻量点 Transformer）是同时期平行工作，也做了 3D RoPE；但 Volt 走得更彻底——**完全没有 U-Net 分层、没有局部窗口**，是 3D 场景理解里第一个纯 vanilla Transformer 编码器 backbone。

## 4. 问题与核心方法

### 4.1 要解决的问题

- **架构隔离**：3D 场景理解还在用带强领域先验的专用 backbone，享受不到 Transformer 生态的成果迁移和软硬件优化（FlashAttention、张量核等）。
- **数据太少**：全局注意力的 vanilla Transformer 假设空间大，在 ScanNet/nuScenes 这种比 ImageNet 小好几个量级的数据上 naive 训练**严重过拟合**（naive 只有 31.0 mIoU，比 PTv3 的 35.2 低 4.2）。
- **普遍担忧**：全全局注意力在 3D 场景上会不会贵到不可用？

### 4.2 核心方法

#### 4.2.1 Tokenization：体素块化（Volumetric Patchification）

输入点云 $\mathcal{X}=\{(\mathbf{p}_n,\mathbf{f}_n)\}_{n=1}^{N}$，$N$ 个点，每点有 3D 坐标 $\mathbf{p}_n\in\mathbb{R}^3$ 和 $C$ 维特征 $\mathbf{f}_n$（如颜色）。

1. **体素化**：用体素尺寸 $\delta$ 把场景规则网格化（室内 2cm、室外 5cm），每个被占用的体素选一个代表点，得到稀疏体素集 $\mathcal{X}_v=\{(\mathbf{z}_m,\mathbf{f}_m)\}_{m=1}^{M}$，其中 $\mathbf{z}_m\in\mathbb{Z}^3$ 是整数体素坐标。
2. **切块**：把稀疏体素网格划分成 $P\times P\times P$ 的非重叠立方 patch（$P=5$）。对每个**非空** patch，把空体素用零特征补齐（densify），完全空的 patch 丢弃。
3. **展平 + 投影**：每个稠密 patch 展平成向量 $\mathbf{u}_t\in\mathbb{R}^{P^3 C}$，再经共享线性层投影到 Transformer 嵌入维 $D$，得到 token $\mathbf{x}_t\in\mathbb{R}^{D}$，组成序列 $\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_T]$。

关键点：token 数 $T$ 随场景大小/点密度变化（不像 ViT 是固定图尺寸），典型场景约 **5000 个 token**（ScanNet、nuScenes），全局注意力的 $\mathcal{O}(T^2)$ 成本在 FlashAttention 下可行（P.5）。

#### 4.2.2 Transformer 编码器

标准 vanilla Transformer：$L$ 个相同的 block（多头自注意力 + MLP + 残差），pre-LayerNorm。**没有**分层 stage、**没有**局部窗口、**没有**领域专用分组算子——只有全局注意力 + MLP。用 FlashAttention 实现全局注意力（可变长度序列的融合 kernel）（P.6）。

#### 4.2.3 3D RoPE 位置编码

用 patchification 后的**离散 3D token 索引**作为位置（它和真实 3D 坐标只差常数因子 $\delta P$，因此跨场景保持度量一致性）。把每头 query/key 向量沿三个空间轴分解，各轴独立做标准 RoPE 旋转：

$$
\begin{aligned}
\tilde{\mathbf{q}} &= \text{concat}\left(\mathcal{R}_{\Theta}(p_x)\,\mathbf{q}_x,\ \mathcal{R}_{\Theta}(p_y)\,\mathbf{q}_y,\ \mathcal{R}_{\Theta}(p_z)\,\mathbf{q}_z\right), \\
\tilde{\mathbf{k}} &= \text{concat}\left(\mathcal{R}_{\Theta}(p_x)\,\mathbf{k}_x,\ \mathcal{R}_{\Theta}(p_y)\,\mathbf{k}_y,\ \mathcal{R}_{\Theta}(p_z)\,\mathbf{k}_z\right)
\end{aligned}
$$

符号说明：

- $\mathbf{q},\mathbf{k}\in\mathbb{R}^{D_h}$：某 token 在位置 $\mathbf{p}=(p_x,p_y,p_z)$ 的单头 query/key 向量；
- $\mathbf{q}_x,\mathbf{q}_y,\mathbf{q}_z$（$\mathbf{k}$ 同理）：按轴切分的子向量，维度分别为 $12,12,8$（**非对称**分配：水平面 x/y 给更多容量，重力对齐的 z 轴变化少，给 8）；
- $\mathcal{R}_{\Theta}(\cdot)$：标准分块对角旋转变换 [6]，由频率 $\Theta$ 参数化；
- $\tilde{\mathbf{q}},\tilde{\mathbf{k}}$：旋转后的 query/key，注意力分数因此只依赖**相对位置偏移**。

作者把 RoPE 与其他方案做了消融：对称分配（12/12/12）降 0.8/1.0；每场景坐标归一化（破坏跨场景度量一致性）降 0.5/1.2；换 Fourier 位置编码大幅降 7.6/4.5（详见实验 5.2）（P.6、P.12）。

#### 4.2.4 轻量 Decoder

- 语义分割：用**单个转置卷积等价操作**把 token 特征上采样回体素分辨率（kernel = patch 大小 $P$，是 tokenization 的"逆"），实现为 linear + voxel shuffle（不依赖稀疏卷积引擎），再对 $M$ 个占用体素的 $D'$ 维特征 $\mathbf{F}\in\mathbb{R}^{M\times D'}$ 接线性分类头。
- 实例分割：沿用常见范式，接 Transformer decoder（如 SPFormer）(P.7)。
- 推理时每个点取所在体素的预测。

#### 4.2.5 数据高效训练配方（核心贡献之二）

1. **强 3D 数据增强**：scene mixing（概率 0.85）、随机裁剪、实例级变换、弹性形变、随机缩放/平移/旋转/翻转、随机点 dropout。
2. **正则化**：DropPath（随机深度，丢弃率随深度线性增加，浅层保留更多）、AdamW 强权重衰减 0.05、label smoothing 0.1、EMA（decay 0.999）、QKNorm。
3. **卷积 teacher 蒸馏**：用同数据训练的 MinkUNet Res16UNet34C 当 teacher，DeiT 风格**硬标签**蒸馏，双头联合损失：

$$
\mathcal{L} = 0.5\,\mathcal{L}_{\mathrm{seg}}\left(\mathbf{W}_{\mathrm{seg}}^{\top}\mathbf{F},\, y_{\mathrm{gt}}\right) + 0.5\,\mathcal{L}_{\mathrm{seg}}\left(\mathbf{W}_{\mathrm{distill}}^{\top}\mathbf{F},\, y_{\mathrm{teacher}}\right)
$$

符号说明：

- $\mathbf{W}_{\mathrm{seg}},\mathbf{W}_{\mathrm{distill}}$：两个并列线性分类头的权重（分割头 + 蒸馏头）；
- $y_{\mathrm{gt}}$：真实标签；$y_{\mathrm{teacher}}$：teacher 的硬预测；
- $\mathcal{L}_{\mathrm{seg}}$：交叉熵 + Lovász 损失（语义分割）。

有意思的是：**teacher 本身性能一直低于 Volt**（这不是"强 teacher 带弱 student"的经典设定），但蒸馏仍然带来 +2.1 mIoU 的提升——teacher 的归纳偏置被"注入"给了 Transformer，训练后 teacher 丢弃，推理零额外开销（P.8）。

4. **多数据集联合训练（Scaling Data）**：室内联合 ScanNet + ScanNet200 + ScanNet++ + ARKit LabelMaker（5,019 张带自动标注的 ARKitScenes 扫描、185 类）；室外联合 nuScenes + Waymo + SemanticKITTI。因为各数据集标签空间不同，用**每数据集一个线性分类头**、共享其余 backbone。注意室内外分开训，遵循领域分组惯例（P.8）。

## 5. 实验

### 5.1 实验设置

- **变体**：Volt-S（ViT-S 配置，23.7M 参数）、Volt-B（ViT-B 配置，87.7M）。
- **通用设置**：voxel $\delta$ = 室内 2cm / 室外 5cm，patch $P=5$（边长 10cm/25cm）；语义分割用 CE + Lovász，实例分割用 Dice + CE；AdamW + 1-cycle LR、weight decay 0.05、label smoothing 0.1、EMA 0.999；8×A100、global batch 16、FP16 + FlashAttention-2；线性层 truncated normal（σ=0.02）初始化（P.8）。
- **数据集**：室内 ScanNet（1613 场景/20 类）、ScanNet200（200 细粒度类）、ScanNet++（956 场景/100 类）、ARKit LabelMaker；室外 nuScenes（32 线 LiDAR/16 类）、Waymo（64 线/22 类）、SemanticKITTI（19 类）。
- **蒸馏 teacher**：MinkUNet Res16UNet34C；实例分割从语义分割 checkpoint 初始化再微调（沿用 [16,86,88-90] 做法）。

### 5.2 训练配方逐项分析（Fig.3 左，ScanNet200 mIoU，P.7）

| 配置 | Volt | PTv3 |
|---|---|---|
| Naive | 31.0 | 35.2 |
| + 强增强 | 32.9 (+1.9) | 35.5 (+0.3) |
| + 强正则 | 34.1 (+1.2) | 35.5 (+0.0) |
| + CNN 蒸馏 | 36.2 (+2.1) | 35.8 (+0.3) |
| + 扩数据 | 38.1 (+1.9) | 36.7 (+0.9) |
| + 扩模型 | 40.0 (+1.9) | 37.5 (+0.8) |

关键观察：**同样的升级序列，Volt 共 +9.0，PTv3 只 +2.3**。正则化和蒸馏对 Volt 的提升远大于 PTv3——说明它们专门缓解了"少归纳偏置 + 有限监督"导致的过拟合。Fig.4 显示数据量从 50%→100%→多数据集递增时，Volt-S 提升曲线比 PTv3 更陡，说明**架构简单反而有更大的数据红利空间**（P.8）。

### 5.3 室内语义分割（Table 1，P.9）

- 单数据集训练：Volt-S 在 ScanNet val 77.2、ScanNet200 val 36.2、ScanNet++ test 49.3，与 PTv3（46.1M）基本打平甚至略优，而参数只有其一半（23.7M vs 46.1M）。
- 联合训练（4 数据集）：Volt-S 达 ScanNet 80.2、ScanNet200 38.1，**超越 5 倍大的 PPT**（124.8M，PTv3-B + 5 数据集含 Structured3D）。
- **Volt-B（联合）刷新全部 SOTA**：ScanNet test 80.5、ScanNet200 test 41.6、ScanNet++ test 49.5。

### 5.4 室外语义分割（Table 2，P.10）

- 单数据集：Volt-S 在 nuScenes 81.0、SemanticKITTI 70.5、Waymo 71.5，**三个数据集 val 全部第一**，且没有引入任何 LiDAR 专属设计（对比 Cylinder3D、SphereFormer、LSK3DNet）。
- 联合训练：Volt-B 达 nuScenes 82.2、SemanticKITTI test 75.2、Waymo 73.4，全面超越 PPT。

### 5.5 室内实例分割（Table 3，P.11）

把 SPFormer 的 MinkUNet backbone 换成 Volt（decoder/损失不动，纯换 backbone 的受控对比）：

- **Volt-S 让 ScanNet200 val 直接 +14.6 mAP50**（33.8→48.4），SOTA。
- Volt-B：ScanNet test 82.7、ScanNet200 test 47.5、ScanNet++ test 54.9，全部新纪录。
- 结论：**只加强底层 3D 表征，收益比设计更花哨的 decoder（OneFormer3D、Relation3D 等）还大**。

### 5.6 功能分割 SceneFun3D（Table 4 左，P.11）

细粒度功能理解（把手、旋钮、按钮等 9 类 affordance）：Volt-B 以 34.5 AP50 大幅超越此前发布的方法（Mask3D-F 18.3、OpenTrack3D 14.0、LERF 12.3）——证明泛化到小部件级细粒度任务依然有效。

### 5.7 消融（Table 4 右、Table 5，P.11-12）

- **位置编码**：Asym. RoPE（默认）36.2/70.5 最好；对称 RoPE 35.4/69.5；Asym + 归一化 35.7/69.3；Fourier PE 28.6/66.0（掉最多）→ RoPE 的相对位置形式、跨场景度量一致性、非对称轴向分配都对。
- **Decoder**：无 decoder 35.0/69.9（最快）；轻量 decoder 36.2/70.5（27/49ms）；**更重的 decoder 反而降到 35.8/69.5**——backbone 表征已经够强，重 decoder 在有限监督下加剧过拟合。
- **Patch 大小**：3³ 精度最高（37.7/71.0）但慢 2.5×（67/113ms）；7³ 快但掉点（33.5/69.6，ScanNet200 掉更多——粗 patch 对小物体/细薄结构敏感）；**5³ 是精度/速度最佳平衡**。

### 5.8 效率（Fig.1，P.2）

全局注意力并不贵：Volt-S 比 MinkUNet 快且峰值内存相当；**Volt-B 比 PTv3-S 快 2×、内存少 38%**（A100 和 H100 均如此）——因为 FlashAttention + 无分层无多尺度结构的简约设计。

## 6. 图表清单

- Fig.1 ScanNet 推理效率对比（Volt-S vs MinkUNet；Volt-B vs PTv3-S）(P.2)
- Fig.2 Volt 架构总览（体素块 token → Transformer 编码器 → 转置卷积上采样 → 线性头）(P.4)
- Fig.3 左：训练配方增量分析表（Volt vs PTv3）；右：有无 CNN 蒸馏的训练曲线 (P.7)
- Fig.4 缩放行为：ScanNet200 50%/100% + 多数据集联合下的 mIoU 曲线 (P.8)
- Fig.5 ScanNet 语义分割定性结果（输入点云 / 预测 / GT）(P.17)
- Fig.6 SemanticKITTI 语义分割定性结果 (P.18)
- Table 1 室内语义分割 mIoU（单数据集 + 联合训练）(P.9)
- Table 2 室外语义分割 mIoU（nuScenes / SemanticKITTI / Waymo）(P.10)
- Table 3 室内实例分割 mAP50（SPFormer + 不同 backbone）(P.11)
- Table 4 左：SceneFun3D 功能分割；右：位置编码消融 (P.11-12)
- Table 5 左：decoder 消融；右：patch 大小消融（含速度）(P.12)

## 7. 快速总结

**核心贡献：**

1. Volt：3D 场景理解第一个**纯 vanilla Transformer backbone**——体素块 token + 全全局注意力 + 3D RoPE，可纯 PyTorch 实现、直接吃 FlashAttention 等生态红利（P.4-6）。
2. 数据高效训练配方（强增强 + DropPath/标签平滑 + 卷积 teacher 蒸馏）解决了少数据下的过拟合，同一配方让 Volt 从 naive 31.0 提到 36.2 mIoU（P.7）。
3. 证明**少归纳偏置架构的缩放优势**：多数据集联合训练下 Volt 受益远大于 PTv3（+9.0 vs +2.3），多基准刷新 SOTA，且效率更高（Volt-B 比 PTv3-S 快 2×、省 38% 内存）（P.2、P.8）。

**主要局限/观察：**

- 当前 3D 数据规模仍是瓶颈：训练配方让 Volt-S 在大数据下开始"容量饱和"（38.1 → 需要 Volt-B 才能继续涨），说明 3D 需要更大监督（P.8）。
- 蒸馏依赖一个卷积 teacher；teacher 比 student 弱但仍有帮助，机制层面（到底注入了什么归纳偏置）没展开分析。
- 重 decoder 反而掉点、3³ patch 精度更高但慢——精度-速度权衡仍需按任务取舍。

**值得借鉴/延伸：**

- "从架构搬 2D 范式，用训练配方补数据"的思路可以直接迁移到 3D 检测、occupancy 预测等任务。
- 3D RoPE 的非对称轴向分配（x/y 12、z 8）+ 离散索引位置的度量一致性设计很干净，可复用到其他 3D Transformer。
- 联合多数据集 + per-dataset head 是低成本扩监督的好模板；ARKit LabelMaker 这种"弱标注大规模数据"的价值被明确量化。

**开源情况：**

- 代码（MIT）：https://github.com/YilmazKadir/Volt —— 官方实现，基于 Pointcept，纯 PyTorch（2026-06 起无 spconv 依赖），预处理脚本覆盖 ScanNet/ScanNet200/ScanNet++/SceneFun3D/nuScenes/SemanticKITTI/Waymo。
- 权重：https://huggingface.co/KadirYilmaz/Volt —— teacher weights 与 checkpoints（`hf download KadirYilmaz/Volt --include "teacher_weights/*.pth"`）。
- 项目页：https://yilmazkadir.github.io/Volt/ ；ECCV 2026，另获 3 个 CVPR 2026 challenge 优胜（ScanNet++ 语义/实例、SceneFun3D 功能分割）。

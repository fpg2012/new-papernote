---
领域: 3D 点云 / 3D 表示学习（计算机视觉）
发表时间: CVPR 2024（arXiv v1 2023-12-15；v2 2024-03-25）
读文章时间: 2026-09-04
标题: Point Transformer V3: Simpler, Faster, Stronger
链接: https://arxiv.org/abs/2312.10035
代码: https://github.com/Pointcept/PointTransformerV3
Zotero: 已入库（key JENL323L，含 PDF 附件）
页码为 PDF 物理页码（即正文印刷页码，标题页为 P.1）
---

# Point Transformer V3: Simpler, Faster, Stronger

## 1. TL;DR

这篇文章面对点云 transformer 主干网络「精度高但效率差、难以靠缩放(scale)受益」的老问题，提出一个核心理念：**模型性能受规模的影响大于受精巧设计的影响**。作者没有在注意力机制内部做创新，而是反其道而行——把点云「序列化」(serialization) 成结构化的 1D 顺序，用高效的 patch 注意力和极简的条件位置编码 **替换掉 KNN 邻居搜索和相对位置编码 RPE**，从而把接受野从 16 扩到 1024 个点还保持高效，最终在 20+ 个室内外下游任务上取得 SOTA（相比上一代 PTv2 推理快 3.3×、省内存 10.2×）。

## 2. 作者与单位

- 吴晓阳 Xiaoyang Wu（香港大学 HKU；上海人工智能实验室）
- 李江 Li Jiang（香港中文大学(深圳) CUHK-SZ）
- 王鹏帅 Peng-Shuai Wang（北京大学 PKU）
- Zhijian Liu（MIT）
- Xihui Liu（HKU）
- 乔宇 Yu Qiao、欧阳万里 Wanli Ouyang（上海 AI Lab）
- 贺通 Tong He（上海 AI Lab，通讯 *）
- 赵恒爽 Hengshuang Zhao（HKU，通讯 *）

## 3. 紧密相关的工作与交叉点

直接相关、构成承继关系的主线是作者自己的一条序列：

- **Point Transformer (PTv1)**（Zhao 等, ICCV 2021）——第一代，用 KNN 构造局部邻域 + 相对位置编码 + vector attention，精度好但极慢。
- **Point Transformer V2 (PTv2)**（Wu 等, NeurIPS 2022）——分组 vector attention + 分区 pooling，把 KNN 的使用砍半，但仍占前向时间 28%。

交叉点上最关键的几支：

- **序列化思路的先行者**：**OctFormer**（八叉树继承 z-order 顺序，用 Shift Dilation 交互，但受限于八叉树结构）与 **FlatFormer**（对 pillar 做 window 式排序，多用于室外检测）。PTv3 站在它们之上，把「把点云排成 1D 序列」这条线推到完整/通用。
- **规模化(scale)的动力**：3D 大规模表示学习 **PPT (Point Prompt Training)** 提出多数据集联合训练，让稀疏卷积 MinkUNet 从 72.2% 涨到 77.0% mIoU。它点明「数据规模」收益大，但这红利此前没被点云 transformer 吃到——正是本文的出发点。
- **对比的对象**：Sparse Convolution（MinkUNet/ConvNeXt）以快和省著称；Swin3D、Stratified Transformer 用更复杂的 RPE 提精度却牺牲速度。

一句话：本文把「序列化点云（来自 OctFormer/FlatFormer）+ 规模优先哲学（受 PPT 启发）+ 点 transformer（PTv2 血统）」三条线拧在一起。

## 4. 问题与核心方法

### 4.1 要解决的问题

点云 transformer 在追求高精度的同时，背着两个效率包袱，导致接受野和模型规模都难以上去（P.3）：

1. **KNN 邻居搜索**：要在不规则点云里找真正的空间近邻，代价高昂，占 PTv2 前向时间约 28%。
2. **相对位置编码 RPE**：图像靠固定网格能预定义相对位置，点云却必须逐对算欧氏距离再映射成 embedding，占前向时间约 26%（两者合计 54%，见 Fig.2）。

传统应对是靠「更精巧的模块」换精度（vector attention、复杂 RPE），但往往牺牲效率，陷入精度-效率的对立。作者主张：**别跟这两个昂贵操作硬磕，干脆放弃置换不变性假设，把点云变成有序结构，用规模把精度找回来**。

### 4.2 核心方法

整体上 PTv3 = 序列化 + patch 注意力 + 简化位置编码，配上 U-Net 骨架。设计哲学是「能删则删、删完能scale，scale 完更强」。

#### 思路概览（P.4）

三个关键取舍（相对 PTv2）：

1. 用**序列化邻居映射**取代 KNN 精确邻居搜索（不追求局部结构的极致保真，换取排序后的规则、可并行的结构）。
2. 用**点乘注意力 + 无重叠 patch**（借鉴图像 window/dot-product attention），取代点云常用的 vector attention / neighborhood attention / shift-window 这类交互复杂、融合难、吃内存的机制。
3. 用**条件位置编码 xCPE** 取代 RPE（去掉慢的 26%），本质上把位置信息交给一个前置稀疏卷积。

#### Step 1 — 点云序列化（Point Cloud Serialization，P.4）

序列化靠**空间填充曲线(space-filling curves)**：一条能遍历高维离散空间里每个点、且大致保持空间近邻的路径。数学上是一个双射函数

$$
\varphi:\;\mathbb{Z}\mapsto\mathbb{Z}^n
$$

其中 $n$ 是维度（点云取 3）。文章用两条代表曲线，并给它们各加一个「换轴顺序」的变体，共四种：**Z-order**、**Hilbert**、**Trans Z-order**、**Trans Hilbert**（trans = 把标准沿 x/y/z 轴的遍历顺序打乱，如先 y 后 x，能捕捉到标准曲线漏掉的局部关系，见 Fig.3）。

序列化编码：把一个点的坐标量化后，用曲线逆映射 $\varphi^{-1}$ 换成一个整数（该点在曲线上的序号），再拼上批次号：

$$
\mathtt{Encode}(\pmb p,\;b,\;g)=(b\ll k)\ \big|\ \varphi^{-1}\!\left(\left\lfloor \pmb p/g \right\rfloor\right)
$$

符号说明：

- $\pmb p\in\mathbb{R}^3$：点的位置坐标。
- $g\in\mathbb{R}$：网格大小（把连续位置投影到离散空间的量化步长）。
- $b\in\mathbb{Z}$：批次索引(batch index)。
- $k$：每个点用 64 位整数记录，尾部落 $k$ 位放序列码，高位移位存批次号 $b$。
- $\lfloor\cdot\rfloor$ 向下取整；$\ll$ 左移位；$\mid$ 按位或；$\varphi^{-1}:\mathbb{Z}^n\mapsto\mathbb{Z}$ 是曲线的逆映射（返回该离散格点的序列序号）。

把所有点按编码排序，序列内相邻的点就大概率在空间里也相邻。实现上作者**不真正物理重排点云，而是记录序列化产生的 index 映射**，这样切换不同序列化 pattern 只需换映射、开销极低。

#### Step 2 — Patch 注意力（Serialized Attention，P.4-5）

在排好序的点序列上做**patch 注意力**：把点按顺序（先 padding 补齐到能被 patch size 整除，padding 从相邻 patch 借点）切成**互不重叠的 patch**，在每个 patch 内部做标准的 **window / dot-product 注意力**。有效与否靠两个设计：

- **Patch grouping（Fig.4）**：重排 + padding 可以合成一次索引操作，高效；随 patch size 增大，注意力感受野在 3D 空间里越铺越广。
- **Patch interaction（Fig.5）**：纯不重叠 patch 会割裂全局信息，需要跨 patch 交互。作者对比了四种：

  - **Shift Dilation**（受 OctFormer 启发）：跨固定步长错开分组，产生「空洞」扩大感受野。
  - **Shift Patch**（借自图像 shift-window）：把 patch 在序列上整体平移，最大化 patch 间交互。
  - **Shift Order**（默认）：在不同注意力块之间**轮换/更换序列化 pattern**，防止模型过拟合单一排序，促进特征融合。
  - **Shuffle Order\***（主打、带 *）：在 Shift Order 基础上对多个序列化 pattern 的顺序做随机打乱，让每层注意力的感受野不局限单一 pattern，泛化更好。

  消融显示 Shift Dilation / Shift Patch 有效但依赖 attention mask、偏慢；**Shift Order + Shuffle Order 再配合全部四种曲线**（Z+TZ+H+TH）又快又准（Tab.2 里 77.3%、61ms）。

#### Step 3 — 条件位置编码 xCPE（P.5-6）

作者观察到 RPE 其实就是一种「大 kernel 稀疏卷积」的变体。与其在注意力内部逐对算距离，不如把位置编码做成一个**前置的带残差连接的稀疏卷积层**（作者叫 enhanced CPE，xCPE），插在注意力层之前。单层 CPE 还不够（配 RPE 能再 +0.5%），xCPE 用稀疏卷积补上这部分，精度追平且只多几毫秒延迟。因为 CPE 在注意力**之前**做，还能吃 FlashAttention 这类算子优化的红利。

#### Step 4 — 网络细节（P.6 + 附录 A.2 / B）

- **骨架**：仍是 U-Net，四段 encoder 深度 [2,2,6,2]、decoder [1,1,1,1]，各段 grid down-stride 均 ×2；用 PTv2 的 **Grid Pooling** 做池化。
- **块结构**：用 **pre-norm**（norm 在算子前，训练稳定），并换用 Layer Normalization（LN）。
- **归一化组合的权衡**：attention 块用 LN、**pooling 层仍保留 BN**（附录 Tab.17 里 [LN block + BN pool] = 77.3 最优；全 LN 反而掉到 75.6）。
- **模型配置**（Tab.15）：encoder channels [64,128,256,512]、num heads [4,8,16,32]、各段 patch size 统一 1024；embedding 通道 32；drop path 0.3。推理用 FlashAttention。
- **数据增强**（附录 Tab.16）：室内外用两套统一 pipeline；有个讨喜点——PTv3 **不需要按范围裁剪点云**（很多现存模型必需）。

## 5. 实验

- **设置**：消融在 ScanNet 语义分割 val 的 mean mIoU 上做，延迟在完整 ScanNet val 上单卡 RTX 4090、batch=1 测平均；`○`=单数据集从头训练，`●`=用 PPT 多数据集联合训练后的模型。实现基于 Pointcept（语义/实例分割）、OpenPCDet（室外检测）。

### 消融（为了说明「哪些设计驱动收益」）

- **序列化 pattern 组合**（Tab.2）：单一 Z-order 只有 74.3；把 Z、TZ、H、TH 全用上并加 Shuffle Order → **77.3、61ms**。多 pattern 几乎不增加开销。
- **patch 交互方式**（Tab.2）：Shift Patch / Shift Dilation 有效但慢（依赖 attention mask）；Shift Order+Shuffle Order 简单高效，效果还更好。
- **位置编码**（Tab.3，ScanNet mIoU/延迟）：APE 72.1(50ms) → RPE 75.9(72ms) → cRPE 76.8(101ms)；而 CPE 76.6(58ms)、**xCPE(CPE+) 77.3(61ms)** —— RPE 类最贵且并非最高，xCPE 又快又最好。
- **patch size（接受野缩放）**（Tab.4）：16→75.0，32→75.6，64→76.3，128→76.6，256→76.8，**1024→77.3**，4096→77.1。够大就涨、但 4096 开始略回落（且 std dev 更大，过大会不稳）。证明「把接受野放大」本身就在涨点。

### 结果对比（为了说明「SOTA + 规模红利」）

- **室内语义分割**（Tab.5/6/19/20）：ScanNet test 从头 77.9、+PPT 79.4（PTv2 仅 74.2）；ScanNet200 test 37.8（PTv2 —）；S3DIS 6-fold 从头 77.7、+PPT 80.8（PTv2 73.5）。**文章原话：从头训练就比 PTv2 高 3.7%（ScanNet）/ 4.2%（S3DIS 6-fold），加 PPT 后差距拉到 5.2% / 7.3%**——规模红利显著。
- **室外语义分割**（Tab.7/21/22）：nuScenes val 80.4/ 从头上比上一代最先进的 SphereFormer 高 2.0%，test 82.7；SemanticKITTI val 高 3.0%、test 74.2；预训练后领先扩到 nuScenes +2.8%、SemKITTI +4.5%。Waymo val mIoU 71.3（PTv2 70.6）。
- **室内实例分割**（Tab.8，统一套 PointGroup 只换 backbone）：ScanNet mAP 从头 40.9 vs MinkUNet 36.0、PTv2 38.3；+PPT 到 42.1（微调再 +1.2 mAP）。说明作为 backbone 的通用性。
- **室外目标检测**（Tab.10，CenterPoint 作检测头）：Waymo 单帧车辆 71.2 mAP / mean L2 70.4，比 SOTA 的 FlatFormer 单帧高 3.3%；多帧设定仍领先 1.0%。
- **数据高效场景**（Tab.9）：ScanNet 限量标注/限量重建下，PTv3 基本全面领先，预训练后再抬一截。

### 效率（核心卖点，Tab.1/11）

- nuScenes 上 PTv3/1024：46.2M 参数，**训练 119ms/3.3G，推理 44ms/1.2G**；对照 PTv2/16：**推理 146ms/12.3G**。也就是说 PTv3 感受野放大到 1024 的同时，速度是 PTv2 的约 3.3×、显存只要约 1/10（正文 P.1 口径 3.3× 速度、10.2× 省内存）。
- 室内 ScanNet（Tab.11）：PTv3 推理 61ms/5.2G，明显快过 Swin3D(456ms)、OctFormer(86ms)、PTv2(191ms)。

### 反常/值得注意的点

- 4096 的 patch size 反而比 1024 略降（Tab.4）：接受野不是越大越好，超出后不升反降且更不稳定——但即便如此仍远超过去 16 的上限。
- pooling 的 BN 去不掉：换成 LN 会掉分，作者推断 BN 在池化时对点云数据分布有稳定作用（Tab.17）。
- post-norm 在这个任务上明显吃亏（72.3 vs pre-norm 77.3），与图像 transformer 里 pre-norm 更好的结论一致（Tab.18）。

## 6. 图表清单

- Fig.1 PTv3 总览：对比 PTv2 在精度/接受野/速度/内存四方面优势 (P.1)
- Fig.2 PTv2 前向各组件延迟 treemap：KNN+RPE 合计 54% (P.3)
- Fig.3 点云序列化：四种空间填充曲线(Z/Hilbert/Trans Z/Trans Hilbert)的三联可视化(曲线/排序/分组 patch) (P.4)
- Fig.4 Patch 分组：(a)重排点云 (b)跨邻 patch 借点 padding 使序列可整除 (P.5)
- Fig.5 Patch 交互：(a)标准 (b)Shift Dilation (c)Shift Patch (d)Shift Order (e)Shuffle Order (P.5)
- Fig.6 整体架构图 (P.6)
- Table 1 室外(nuScenes)训练/推理效率对比，含多种感受野 (P.3)
- Table 2 序列化 pattern × patch 交互组合消融（mIoU/延迟） (P.6)
- Table 3 位置编码对比：APE/RPE/cRPE/CPE/xCPE (P.6)
- Table 4 patch size 缩放与标准差 (P.6)
- Table 5 室内语义分割 ScanNet/ScanNet200/S3DIS (P.7)
- Table 6 S3DIS 6-fold 分区域明细 (P.7)
- Table 7 室外语义分割 nuScenes/SemKITTI/Waymo (P.7)
- Table 8 室内实例分割 ScanNet/ScanNet200 (P.7)
- Table 9 数据高效(限量重建/限量标注)对比 (P.7)
- Table 10 Waymo 目标检测（单/多帧） (P.8)
- Table 11 室内(ScanNet)模型效率 (P.8)
- Table 12 室内语义分割训练设置（scratch vs 联合训练） (P.9)
- Table 13 室外语义分割训练设置 (P.9)
- Table 14 实例分割与目标检测设置 (P.9)
- Table 15 模型配置明细 (P.10)
- Table 16 数据增强设置 (P.10)
- Table 17 归一化层消融（block×pooling） (P.10)
- Table 18 块结构消融（traditional/post/pre-norm） (P.10)
- Table 19 ScanNet V2 语义分割全历史对比（含预训练方法） (P.11)
- Table 20 S3DIS 语义分割全历史对比 (P.11)
- Table 21 SemanticKITTI 语义分割对比 (P.12)
- Table 22 NuScenes 语义分割对比 (P.12)

## 7. 快速总结

**核心贡献**
1. 把「规模优先」哲学落到点云 backbone 上：用简单高效换取可扩展性，靠放大数据/感受野把精度找回来，与近年 3D 大规模表示学习呼应。
2. 系统提出并验证**点云序列化**（四种空间填充曲线 + Shift/Shuffle Order 跨块换序），用序列内 patch 注意力替代 KNN，兼顾结构与效率。
3. 用极简**条件位置编码 xCPE** 干掉昂贵的 RPE，配 pre-norm/LN 等现代结构，把接受野从 16 拉到 1024，成为通用、更快更省的新 SOTA backbone。

**主要局限**
- 回归点乘注意力后收敛更慢、深度难再堆（作者归因于点积+softmax 带来的 "attention sinks"，P.9）。
- 序列化牺牲了严格的局部保真度，极端不规则/依赖精确近邻的场景可能有损。
- 放大 patch 并非无上限（4096 开始回落）；且缩放红利的证明依赖大规模联合训练/算力。

**值得借鉴 / 可延伸**
- 「别在昂贵操作上硬优化，先想能不能删掉、删了能否靠规模补」的反直觉设计方法论。
- 点云序列化可移植到 2D 图像 / 多模态 1D 序列编码（作者明确点出可做 2D-3D 联合预训练），是一套相对通用的框架。
- 附赠的工程红利：无需按范围裁剪点云、能直接吃 FlashAttention，是「易用性」设计的范例。

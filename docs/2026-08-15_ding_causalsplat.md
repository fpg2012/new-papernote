---
领域: 3D 重建 / 神经渲染（3DGS 场景理解与推理分割）
发表时间: 2026-08-11（arXiv v2）
读文章时间: 2026-08-15
标题: CausalSplat: Towards Comprehensive Hierarchical Reasoning in 3D Gaussian Splatting
链接: https://arxiv.org/abs/2608.11150
Zotero: 已入库 [zotero://select/library/items/VK2PSG2X](zotero://select/library/items/VK2PSG2X)
页码为 PDF 页码
---

# CausalSplat: 面向 3DGS 的层次化推理分割

## 1. TL;DR

现有 3DGS 开放词汇/指代理解只能处理显式名词查询，无法解析隐含意图、复杂空间约束和常识推理。本文定义了新任务 **Reasoning 3D Gaussian Segmentation（3D 高斯推理分割）**，建了两个基准 Causal-LERF（2D 像素级）和 Causal-ScanNet（3D 点级），系统评估常识、空间、功能（affordance）、预测/反事实四层推理能力；并提出了 CausalSplat 框架——用 LLM/VLM + 3D 语义场景图，把"显式结构感知"和"隐式逻辑推理"解耦，在新基准上大幅领先 SOTA（Causal-LERF 47.0 vs 第二名 23.6 mIoU；Causal-ScanNet 14.9 vs 5.1），且不专门设计也能在标准指代/开放词汇分割任务上达到 SOTA。

## 2. 作者与单位

- 丁嘉宇 Jiayu Ding*（北京大学深圳研究生院；广东省超高清沉浸式媒体技术重点实验室）
- 宋美璐 Meilu Song*（华北电力大学计算机系，保定）
- 陈赟 Yun Chen*（湖南大学计算机与电子工程学院）
- 高伟 Wei Gao（北京大学深圳研究生院；同上实验室）
- 李革 Ge Li†（北京大学深圳研究生院；同上实验室，通讯作者）

（* 共同一作；† 通讯作者）

## 3. 紧密相关的工作与交叉点

本文站在三条线交叉点上：

1. **3DGS 场景理解**：像素级（LangSplat、Feature3DGS——把 2D 视觉基础模型特征蒸馏进 3D 特征场，常有多视图不一致）和点级（OpenGaussian、DrSplat——把 2D mask/CLIP 特征提升到 3D 高斯中心）。R3DGS 类方法（ReferSplat、ZeroSplat）用复杂语言描述定位目标，但仍依赖显式名词/形容词。
2. **推理分割（2D）**：LISA 首创（LVLM + SAM），PixelLM、LLM-Seg、LLaVASeg（CoT 提示）、VISA（视频）等；3D 域主要是点云表示：PARIS3D、Reasoning3D（孤立物体部件级）、SegPoint、Reason3D。
3. **3DGS 上的零星推理工作**：3DAffordSplat 只做合成场景的功能推理；REALM 用 MLLM agent 做有限常识/空间推理。两者都是单一维度，缺系统性框架。

**差异点**：现有工作各管一维，本文提出四层递进推理分类（常识/空间/功能/预测反事实）并配基准，且用"场景图承载显式结构 + VLM 承载隐式推理"的方式解决直接特征对齐的两个瓶颈：CLIP 类模型的 bag-of-words 效应导致缺乏拓扑/相对几何编码；视觉-语言对齐抓不住隐含因果逻辑和功能描述。

## 4. 问题与核心方法

### 4.1 要解决的问题

给定 3D 高斯场景和一条带隐含意图/逻辑的自然语言指令（如"我刚洗完手，请把水池旁柜子下挂着的那个擦手的东西递给我"），模型要解析指令并准确分割目标物体。这需要从基础空间感知到物理因果推理的多级认知能力（P.3）。

直接做法（R3DGS 式：给每个高斯赋语义特征、与文本特征对齐）有两个瓶颈（P.1-2）：
1. **缺结构化空间感知**：CLIP 类特征对齐受 bag-of-words 效应限制，编不进物体间拓扑与相对几何关系，无法解析方向、包含等空间约束。
2. **缺隐式推理**：视觉-语言对齐只抓显式属性，抓不住"切牛排的工具 → 刀"这类抽象因果映射。

### 4.2 核心方法

**思路**：把推理过程拆成"显式结构感知"（交给场景图）和"隐式逻辑推理"（交给 LLM/VLM）。场景图显式建模物体间的拓扑/空间关系，补上特征场的空间短板；LLM 用先验知识把隐含查询解析成可执行的结构化意图，补上逻辑短板。三个模块（P.2-3）：语义场构建 → 多模态场景图构建 → 基于场景图的多模态推理。

**模块 1：语义场构建（Sec. 4.1）**——把多视图 2D 分割 mask 提升为 3D 语义特征场，分三步：

*空间加权特征提取*：给每个 3D 高斯点一个语义特征向量 $\mathbf{f}_i \in \mathbb{R}^C$，用与颜色渲染相同的 alpha 混合把特征 splat 到图像平面（P.3）：

$$
\mathbf{F}_{x,y}=\sum_{i\in\mathcal{N}}\mathbf{f}_{i}\,\alpha_{i}\prod_{j=1}^{i-1}(1-\alpha_{j}),
$$

$\mathcal{N}$ 是按深度排序的、覆盖像素 $(x,y)$ 的高斯集合，$\alpha_i$ 是第 $i$ 个高斯的不透明度。SAM 给每个视图 $v$ 出一组 2D mask $\mathcal{M}_v$。mask 边界有固有语义噪声，所以用**空间加权**：对 mask $M_{pos}$ 算几何中心 $(c_x,c_y)$，像素 $(x,y)$ 的权重为

$$
W_{x,y}=\omega_{\min}+\left(1-\omega_{\min}\right)\left(1-\frac{d_{x,y}}{d_{max}+\epsilon}\right),
$$

$d_{x,y}$ 是像素到中心的欧氏距离，$d_{max}$ 是 mask 内最大距离，$\omega_{\min}$ 是最小权重阈值（中心区域权重高、边界噪声被抑制）。加权聚合得 mask 平均语义特征 $\mathbf{p}$：

$$
\mathbf{p}=\mathrm{Norm}\Big(\sum_{(x,y)\in M_{pos}}W_{x,y}\cdot\mathbf{F}_{x,y}\Big),
$$

$\mathrm{Norm}$ 是 $\ell_2$ 归一化。再算 $\mathbf{p}$ 与全局特征场的余弦相似度图 $S$ 供后续使用。

*对比特征优化*：端到端对比损失 $\mathcal{L}_{total}=\mathcal{L}_{pos}+\mathcal{L}_{neg}$（P.4）。正对齐项用 $W_{x,y}$ 衰减边界噪声，把目标区域特征向 mask 平均特征聚拢：

$$
\mathcal{L}_{pos}=\frac{\sum_{(x,y)\in M_{pos}}W_{x,y}(1-S_{x,y})^{2}}{\sum_{(x,y)\in M_{pos}}W_{x,y}},
$$

$S_{x,y}$ 是像素 $(x,y)$ 处 $\mathbf{p}$ 与特征场的余弦相似度。负排斥项用 margin $m$ 惩罚相似度超过 $m$ 的负样本：

$$
\mathcal{L}_{neg}=\frac{1}{|M_{neg}|}\sum_{(x,y)\in M_{neg}}\ell_{neg}(S_{x,y}),\qquad \ell_{neg}(S_{x,y})=\begin{cases}(S_{x,y}-m)^{2}, & S_{x,y}>m\\ 0, & \text{otherwise}\end{cases}
$$

负样本占图像大部分，全用会造成严重类别不平衡，故用指示函数做**动态采样**：

$$
\mathbb{I}_{j}=\begin{cases}1, & \text{if } \eta_{j}<r \text{ or } S_{j}>\tau\\ 0, & \text{otherwise}\end{cases}
$$

$\eta_j\sim\mathcal{U}(0,1)$ 是均匀采样因子，$\tau$ 是难负样本阈值，采样率 $r=N_{pos}/N_{neg}$ 是正负样本像素数之比（保留难负样本、按比例随机采简单负样本）。

*3D 实例分配*：所有视图的 mask 平均特征收集后，用 HDBSCAN 密度聚类成 3D 实体簇 $O_k$，实体代表特征 $C_k$ 取簇内特征均值并归一化：

$$
\mathbf{C}_{k}=\mathrm{Norm}\Big(\frac{1}{|O_k|}\sum_{\mathbf{p}\in O_k}\mathbf{p}\Big).
$$

对每个 3D 高斯点 $j$，算它特征与所有实体代表特征的余弦相似度 $e_{j,k}=\mathbf{f}_j\cdot\mathbf{C}_k^\top$，按硬分配原则取最高分对应的实体 ID：$\mathcal{P}_i=\{j\in\mathcal{P}\mid \arg\max_k(e_{j,k})=i\}$。

**模块 2：语义场景图构建（Sec. 4.2）**——把 3D 实体变成结构化多模态图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$：

*语义节点*：每个实体 $n_i\in\mathcal{V}$ 带多模态描述子 $\mathcal{M}_i=\langle c_i,\mathbf{b}_i,\mathcal{I}_i,\mathcal{A}_i\rangle$：空间质心 $\mathbf{c}_i\in\mathbb{R}^3$（实体高斯点均值坐标）；轴对齐包围盒跨度 $\mathbf{b}_i=\max(\mathcal{P}_i)-\min(\mathcal{P}_i)\in\mathbb{R}^3$；关联的 2D 分割 mask 集 $\mathcal{I}_i$；以及 VLM 从 2D mask 提取的结构化属性元组 $\mathcal{A}_i=\{\mathcal{T}_i,\mathcal{X}_i,\mathcal{F}_i\}$（内在类别、视觉特征、交互功能）。

*尺度自适应边*：开放世界尺度差异大，固定距离阈值不可靠。以垂直关系为例：两实体 $n_i,n_j$ 质心间 Z 轴位移 $\Delta Z=z_i-z_j$ 必须占质心距离 $d_{i,j}$ 的较大比例（$|\Delta Z|/d_{i,j}>\tau$），且水平投影距离小于动态容差：

$$
\mathcal{D}_{limit}^{xy}=\min\Big(\max\big(0.55\sqrt{\bar{b}_x^2+\bar{b}_y^2},\,0.08\mathcal{S}_{xy}\big),\,0.35\mathcal{S}_{xy}\Big),
$$

$\bar{b}_x,\bar{b}_y$ 是两实体包围盒沿对应轴的平均跨度（$\bar{b}_x=(b_{i,x}+b_{j,x})/2$），$\mathcal{S}_{xy}=\sqrt{(\Delta X_{scene})^2+(\Delta Y_{scene})^2}$ 是整个场景的水平面对角线跨度（由所有有效节点质心坐标极值算出）。容差同时平衡物体大小与场景范围（P.4-5）。

**模块 3：多模态推理（Sec. 4.3）**——VLM（Qwen3-VL-30B-A3B-Instruct）三段式推理管线（P.5-6）：
1. **指令解析**：解析指令 $Q$ 语义；只含内在特征就跳到决策输出；检测到显式空间关系词就触发拓扑推理。
2. **拓扑推理**：在图 $\mathcal{G}$ 里定位指令描述的锚点节点 $v_{anchor}$，沿有向边 $E$ 做结构化搜索。
3. **决策输出**：融合语义与拓扑约束，输出最终目标节点。

## 5. 实验

- **数据**：Causal-LERF 与 Causal-ScanNet 共 231 条推理指令、14 个真实室内 3D 场景。指令分布：空间推理 18.2%、常识推理 48.5%、功能推理 16.0%、预测/反事实推理 17.3%。全部指令经四位专业标注员从物理/逻辑一致性多角度审核，全部通过才保留（P.3）。指标：Causal-LERF 用 2D mIoU，Causal-ScanNet 用 3D mIoU。
- **实现**：SAM 提 2D mask、HDBSCAN 聚类；$\omega_{min}=0.4$、对比 margin $m=0.03$、难负阈值 $\tau=0.5$、垂直边阈值 0.3；所有特征 $\ell_2$ 归一化；VLM 用 Qwen3-VL-30B-A3B-Instruct（P.5-6）。

**推理分割（Table 2-3, P.6）**：
- Causal-LERF：Ours 47.0 mIoU，第二名 LUDVIG 仅 23.6；Teatime 上 68.4、Figurines 46.9、Waldo 46.5、Ramen 26.2。开放词汇/指代方法缺推理能力（OpenGaussian 6.3、ReferSplat 10.0），推理类 REALM 也只有 11.4——说明现有方法面对隐含意图集体失效。
- Causal-ScanNet：Ours 14.9，几乎是第二名 LUDVIG（5.1）的三倍；开放词汇方法 1.4~5.1。大点云密集干扰物场景下常识推理极难，差距更悬殊。

**指代分割泛化（Table 4, P.8）**：Ref-LERF 上 Ours 36.1 mIoU，超过前 SOTA ReferSplat（29.2）6.9 点；除 Ramen（28.2 vs 35.2）外各场景都领先。非专门设计仍 SOTA，说明框架泛化性好。

**开放词汇分割泛化（Table 5, P.8）**：LERF-OVS（LangSplat 重标注）上，点级方法中 Ours 51.3 mIoU 最佳（第二 LUDVIG 50.4）；像素级最佳是 3DVLGS 62.0（注意：像素级与点级是不同评估设置，作者只声称点级 SOTA）。Figurines 上 75.3 很高，但 Waldo 只有 30.1。

**消融（Table 6, P.8）**：
- 语义场构建：空间加权 47.0 > 平均池化 35.7 > 随机加权 34.9——结构化特征聚合是必要的，边界噪声影响巨大。
- 场景图：纯文本图 40.1；只加图像反而降（37.2）；只加边几乎无增益（40.2）；图像+边联合（完整多模态图）47.0——两种模态必须联合建模。
- VLM 提示：去掉指令解析 40.5（简单查询上冗余搜索引入错误）；去掉拓扑推理 42.2（复杂空间任务失败）；完整管线动态路由 47.0。

## 6. 图表清单

- Fig.1 任务范式总览：左为 3DGS 分割从开放词汇→指代→隐式意图推理的演进，中为四层推理分类，右为 CausalSplat 相对 SOTA 的增益 (P.1)
- Fig.2 方法总览：(a) 语义场构建（SAM mask → 空间加权特征 → 对比优化 → HDBSCAN 聚类为 3D 实体）；(b) 场景图构建与三段式推理 (P.3)
- Fig.3 Causal-LERF 定性对比（常识/多跳空间查询示例）(P.5)
- Fig.4 Causal-ScanNet 定性对比（功能/空间约束查询，Dr.Splat 等基线过分割）(P.6)
- Table 1 3DGS 理解/推理数据集对比（任务类型、四层推理能力覆盖）(P.3)
- Table 2 Causal-LERF 定量（2D mIoU，7 方法）(P.5)
- Table 3 Causal-ScanNet 定量（3D mIoU，5 方法）(P.5)
- Table 4 Ref-LERF 指代分割（7 方法）(P.7)
- Table 5 LERF-OVS 开放词汇分割（13 方法，像素/点级分列）(P.7)
- Table 6 关键组件消融（语义场构建 / 场景图 / VLM 提示）(P.7)

## 7. 快速总结

**核心贡献**
1. 正式定义新任务"Reasoning 3D Gaussian Segmentation"并给出四层递进推理分类（常识/空间/功能/预测反事实），填补 3DGS 点级推理基准空白。
2. 建了两个基准（Causal-LERF、Causal-ScanNet），标注流程严格（LLM 生成 + 四标注员物理/逻辑一致性审核）。
3. CausalSplat 用"场景图（显式结构）+ VLM（隐式推理）"解耦，解决特征直接对齐的两个瓶颈；在新基准上以近 2~3 倍优势超过现有方法。

**主要局限**
1. 依赖 SAM + HDBSCAN 的实例提取质量：实例分配错了，图就错了，推理无从谈起；论文没讨论重遮挡/小物体的实例失败。
2. 场景图构建成本不低（每个实体要用 VLM 提取属性元组，且要逐视图 SAM）；查询时还要整图输入 VLM。
3. 消融与主实验都用同一批数据，基准是自建的，缺少与外部推理基准的交叉验证；Ramen 场景（26.2）明显低于其他场景，未展开解释。
4. 预测/反事实推理只占 17.3% 指令，且论文没有专门分析这一层最难能力的失败模式。

**可借鉴/可延伸**
1. "显式结构感知与隐式逻辑推理解耦"是个好范式：特征场管"是什么"，图管"在哪/什么关系"，LLM 管"意味着什么"。
2. 尺度自适应边的构造（用物体尺寸 + 场景跨度动态定容差）可复用到其他 3D 场景图工作。
3. 空间加权 + 对比优化 + HDBSCAN 的语义场构建管线，独立于推理部分也有价值，可直接用于开放词汇分割。

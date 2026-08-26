---
领域: "3D 重建 / 神经渲染（稀疏视角 3DGS 恢复）"
发表时间: "2026-08-10（arXiv v1）"
读文章时间: "2026-08-15"
标题: "TRACE-GS: On-Policy Trajectory Distillation with Privileged Geometric Conditioning for Sparse-View 3DGS Restoration"
链接: "https://arxiv.org/abs/2608.10286"
Zotero: "已入库 [zotero://select/library/items/CH7DF6JB](zotero://select/library/items/CH7DF6JB)"
页码说明: "PDF 页码"
---

# TRACE-GS: 用特权几何条件做 on-policy 轨迹蒸馏，恢复稀疏视角 3DGS

## 1. TL;DR

稀疏视角（3/6/9 张图）重建的 3DGS 渲染质量差，现有扩散修复方法有一个根本缺陷：**训练时监督的是"独立加噪的状态"，推理时模型走的是自己预测驱动的 rollout，两者覆盖的状态不一致**（off-policy mismatch），而且稀疏视角几何不约束，偏差会沿 rollout 累积。TRACE-GS 提出 on-policy 轨迹蒸馏：训练时用一个**稠密视角**重建的 3DGS 渲染作条件，让教师（与学生在同一架构、共享骨干）在学生自己 rollout 经过的每个状态上给出目标，对齐去噪方向和跨视图检索响应。稠密几何只存在于训练阶段，属于 LUPI（利用特权信息学习）设定；部署时只保留稀疏条件的学生。

## 2. 作者与单位

- Linlian Jiang（Concordia University；Mila – Quebec AI Institute）
- Yuchen Xi（上海交通大学）
- Sadman Rakib Pinon（Concordia University；Mila）
- 杨睿刚 Ruigang Yang（上海交通大学）
- Yang Wang（Concordia University，共同通讯）
- 左欣欣 Xinxin Zuo（Concordia University，共同通讯）

## 3. 紧密相关的工作与交叉点

本文站在三条线交叉点上：

1. **稀疏视角 3DGS 的扩散先验恢复**：Difix3D+ 用单图扩散逐视图恢复；GenFusion、GSFixer 用视频扩散联合恢复序列以保持跨视图一致。这三者都是"把伪观测喂回 3DGS 优化"，但监督都只在**独立加噪状态**上，rollout 状态无人管（P.2）。本文指出这是训练—推理不一致（exposure bias 的一种）。
2. **On-policy 蒸馏**：模仿学习中 on-policy 监督（DAgger 系），近年用于自回归 LLM 蒸馏（如 Agarwal et al. 2024）、少步扩散轨迹匹配。本文首次把它引入稀疏视角 3DGS 恢复——这个场景下从稀疏条件本身拿不到可靠目标，需要特权几何提供。
3. **LUPI / 广义蒸馏**（Vapnik & Vashist 2009；Lopez-Paz et al. 2015）：利用训练期有、部署期无的信息。常见做法是"大教师→小学生"的容量不对称；本文的教师与学生**架构相同、容量相同，只有几何条件不对称**——教师看到更可靠的稠密几何，这是"几何不对称"的 LUPI 新形式。

**差异点**：与 GSFixer 相比，TRACE-GS 不改变恢复架构，而是改变监督方式（在哪类状态上给监督、目标从哪来）；与普通蒸馏相比，目标不是让学生模仿教师的一般行为，而是在学生自己会访问的 rollout 状态上给出稠密几何条件下的去噪方向。

## 4. 问题与核心方法

### 4.1 要解决的问题

两个耦合的问题（P.1-2）：

1. **off-policy 状态失配**：扩散模型训练时在"干净目标加噪"的状态上监督；推理时模型从纯噪声出发、用自己的预测逐步去噪，会遇到训练目标从未覆盖的自生成状态（对应模仿学习里的 exposure bias）。
2. **几何诱导的偏差累积**：稀疏视角重建的 3DGS 几何不可靠，渲染有伪影（floaters、破碎几何、纹理模糊），从一开始就把去噪方向带偏；沿 rollout 偏差越积越大。

两者相互作用：偏的方向把学生推向未覆盖状态，在那里预测偏差进一步放大。

### 4.2 核心方法

**思路**：需要一个既（i）提供可靠去噪方向、又（ii）在学生实际经过的状态上监督的训练信号。on-policy 蒸馏天然满足（ii）；而（i）靠"特权几何"解决——训练场景通常有额外视角，用全部视角重建的 3DGS 渲染几何更可靠、伪影更少，用它作教师条件就能给出偏差更小的方向，且无需改架构。

**预备（稀疏视角 3DGS + 扩散恢复）**：3DGS 从 $K$ 张带位姿的图像 $\mathcal{V}_s=\{(I_i,P_i)\}_{i=1}^K$（$K\in\{3,6,9\}$）重建出高斯集合 $\mathcal{G}$；沿目标轨迹 $\mathcal{P}=\{P_n\}_{n=1}^L$ 渲染出新视角序列 $R=\{\tilde{I}_n\}_{n=1}^L$。扩散恢复用视频扩散模型在去噪时间表 $t_M>\cdots>t_0$ 上迭代：

$$
z_{t_{i-1}}=\mathcal{T}_{i}\big(z_{t_{i}},\,v_{\theta}(z_{t_{i}},t_{i},R)\big),
$$

$\mathcal{T}_i$ 是调度器更新，$v_\theta$ 是速度（velocity）预测器。解码出的序列 $\widehat{I}$ 与目标位姿配对，作为伪观测与 $\mathcal{V}_s$ 一起监督 3DGS 的迭代优化（P.2）。

**几何不对称的 LUPI 公式（Sec. 3.2）**：设 $\mathcal{V}_s \subset \mathcal{V}_d$（稀疏 ⊂ 稠密视角集）。分别用两个视角集重建 3DGS，沿同一轨迹 $\mathcal{P}$ 渲染出配对序列 $R_s$ 和 $R_d$：

$$
\mathcal{V}_{d}\xrightarrow{\text{3DGS}}R_{d}\xrightarrow{\text{encode}}c_{d}\xrightarrow{\text{restore}}\widehat{I}_{T} \qquad \mathcal{V}_{s}\xrightarrow{\text{3DGS}}R_{s}\xrightarrow{\text{encode}}c_{s}\xrightarrow{\text{restore}}\widehat{I}_{S}\xrightarrow{\text{update}}\mathcal{G}'
$$

$c_b$（$b\in\{s,d\}$）是 $R_b$ 隐式携带的几何。$R_d$ 由更多视角拟合的 3DGS 渲染，几何更可靠，构成**特权信息**：训练期可用、部署期不可用。稠密路径提供特权监督 $\widehat{I}_T$，只有稀疏输出 $\widehat{I}_S$ 在部署时更新 3DGS。

**教师—学生角色（Sec. 3.3）**：在同一个速度预测器 $v_{\theta,\phi}$ 里定义两个角色，共享冻结骨干 $\theta$ 和可训练 LoRA $\phi$。两者接收相同的 $z_t$（$\widehat{I}_T$ 的潜在编码加同一共享噪声 $\epsilon$ 得到），分别以 $R_d$ 和 $R_s$ 为条件：

$$
v_{T}=v_{\theta,\phi}(z_{t},t,R_{d})\;\;(\text{教师，特权}), \qquad v_{S}=v_{\theta,\phi}(z_{t},t,R_{s})\;\;(\text{学生，标准}).
$$

**特权方向对齐（warm-up）**：在独立加噪状态上对齐两条路径的单步预测：

$$
\mathcal{L}_{\mathrm{align}}=\mathbb{E}_{(R_s,R_d),t,\epsilon}\left[\left\|v_{S}-\mathrm{sg}[v_{T}]\right\|_{2}^{2}\right],
$$

$\mathrm{sg}[\cdot]$ 是 stop-gradient——对齐是单向的：把 $v_S$ 拉向稠密条件下的方向，但不反向传播 $v_T$。两条路径同时在 $\widehat{I}_T$ 上做标准 flow-matching 监督 $\mathcal{L}_{\mathrm{warm}}=\mathcal{L}_{\mathrm{fm}}^S+\mathcal{L}_{\mathrm{fm}}^T+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$，把两个预测都锚定到伪目标速度，防止"条件无关的捷径"。因为 $\phi$ 共享，$v_T$ 每次更新重算，是**在线目标**（P.3）。

**On-policy 轨迹蒸馏（Sec. 3.4）**：方向对齐只覆盖独立采样状态，学生递归 rollout 依然无监督，两个局限依然在：学生访问的状态无直接监督、几何误差沿 rollout 累积。解决：在学生访问的每个状态上查询冻结教师。

学生 rollout（$\phi_T$ 冻结，$\phi_S$ 可训练；时间网格 $\{t'_i\}_{i=0}^{N}$ 每次迭代从部署时间表重采样）：

$$
u_{i}^{S}=v_{\theta,\phi_{S}}(z_{t'_i}^{S},t_{i}',R_{s}),\qquad z_{t'_{i-1}}^{S}=\mathrm{sg}\Big[\mathcal{T}_{i}(z_{t'_i}^{S},u_{i}^{S})\Big],
$$

从 $z_{t'_N}^{S}\sim\mathcal{N}(0,\mathbf{I})$ 出发滚 $i=N,\ldots,1$。访问过的轨迹记为 $\mathcal{Z}_S=\{(z_{t'_i}^{S},t_{i}')\}_{i=1}^{N}$。

教师监督：在每个学生访问状态上，冻结教师直接查询（不做自己的 rollout）：

$$
u_{i}^{T}=v_{\theta,\phi_{T}}(z_{t'_i}^{S},t_{i}',R_{d}),\qquad i=1,\ldots,N.
$$

轨迹蒸馏损失：

$$
\mathcal{L}_{\mathrm{traj}}=\mathbb{E}_{(R_s,R_d),z_{t'_N}^{S}}\left[\frac{1}{N}\sum_{i=1}^{N}\left\|u_{i}^{S}-\mathrm{sg}[u_{i}^{T}]\right\|_{2}^{2}\right].
$$

梯度经每个访问状态上的 $u_i^S$ 流向 $\phi_S$，但不穿过被 detach 的调度器转移——所以学生每步都在真实去噪路径上被拉向教师（P.3-4）。

**On-policy 检索对齐（P.4）**：两条路径都关注同一批时间步不变的参考视图 token $T_{ref}$（GSFixer 的机制），但各自 LoRA 投影可能产生不同检索响应；速度匹配不直接约束它。定义注意力响应：

$$
E^{b}=T_{ref}\,W_{K}^{b},\qquad F_{i}^{b}=\mathrm{Softmax}\Big(\frac{H_i^b W_Q^b (E^b)^\top}{\sqrt{d}}\Big)W_{V}^{b},
$$

$b\in\{S,T\}$，$H_i^b$ 是隐藏状态，$W_{Q,K,V}^b$ 含 $\phi_b$。用 $\bar{F}_i^b=\mathrm{Norm}(F_i^b)$（逐 token $\ell_2$ 归一化）后对齐：

$$
\mathcal{L}_{\mathrm{ret}}=\mathbb{E}_{(R_s,R_d),z_{t'_N}^{S}}\left[\frac{1}{N}\sum_{i=1}^{N}\left\|\bar{F}_{i}^{S}-\mathrm{sg}[\bar{F}_{i}^{T}]\right\|_{2}^{2}\right].
$$

总目标 $\mathcal{L}=\mathcal{L}_{\mathrm{traj}}+\lambda_{\mathrm{ret}}\mathcal{L}_{\mathrm{ret}}$。

**部署期 3DGS 细化**：训练后只留 $\phi_S$。对 $K$ 张稀疏视图重建的场景，学生恢复其伪影渲染 $R_s$（无稠密重建、无教师），恢复结果作为伪观测与 3DGS 优化交替 $N_r$ 轮（Algorithm 1, P.4）。

## 5. 实验

- **设置**：冻结预训练视频扩散恢复骨干（GSFixer/CogVideoX 系），训练 rank-32 LoRA，480×720 分辨率；AdamW，lr 1e-5，$\lambda_{align}=1$，$\lambda_{ret}=0.15$；每次迭代从 50 步部署时间表重采样 10 步 rollout；部署期交替恢复与 3DGS 优化 $N_r=3$ 轮（单 A100）。训练数据：DL3DV 112 场景，每场景 $K\in\{3,6,9\}$ 稀疏重建 + 全视角稠密重建，沿共享轨迹渲染 150 对 clip；学生输入 $R_s$ 来自 7K/17K 迭代的稀疏渲染，$R_d$ 来自 30K 迭代稠密渲染（P.4-5）。
- **评测**：DL3DV-Benchmark（同域，28 场景与训练域同分布但不重叠）、Mip-NeRF 360、NeRF-Busters（跨域），指标 PSNR/SSIM/LPIPS。

**同域（Table 1, P.5）**：DL3DV 上 3/6/9 视角 TRACE-GS 的 PSNR 全面最优（16.92/19.46/21.07），3 视角时三个指标全优。对 vanilla 3DGS 的 PSNR 增益随视角变少而增大：9 视角 +2.02、6 视角 +2.35、3 视角 +3.20 dB——与"越稀疏、几何越不可靠、特权指导越值钱"的设计预期一致。对比 GSFixer（3 视角 16.21）：+0.71 dB。

**跨域（Table 2, P.6）**：Mip-NeRF 360 与 NeRF-Busters 上每个稀疏度 PSNR/SSIM 最优；3 视角三指标全优；Mip-NeRF 360 的 6 视角也三指标全优。3 视角比 GSFixer 高 0.73 dB（Mip-NeRF 360）和 0.60 dB（NeRF-Busters）。定性上（Fig. 3-4）：3 视角时基线丢粗几何（桌腿断裂、皮卡丘耳朵塌陷），稠密视角时瓶颈转向细细节（GSFixer 过平滑花朵、沙发轮廓、消防栓表面），TRACE-GS 两者都保住了。

**消融（Table 3, P.7）**——核心是隔离"可靠方向"和"on-policy 状态"两个因素：
- **特权几何的价值**：教师视角 3→12→18→全部 单调提升（DL3DV PSNR 18.36→18.80→18.92→19.15），比无教师基线 (A) 18.64 高 0.51 dB；但**非特权**三视角教师 (B) 18.36 反而低于基线 (A)——蒸馏本身没有额外几何证据就没用。
- **On-policy 监督的价值**：(E) 与 (F) 教师相同（全视角）、预算相同，唯一区别是查询的状态：独立采样 (E) vs 学生访问 (F)。on-policy 全面更优（DL3DV 18.98→19.15；Mip-NeRF 17.53→17.84）。
- **检索对齐**：去掉 $\mathcal{L}_{ret}$ 的 (F⁻) 19.09 仍优于 (E) 18.98，检索对齐再加一点。
- **轨迹行为（Fig. 6）**：两个变体早期去噪步接近，后半程分离——off-policy 变体平台化、on-policy 持续提升。这符合"误差在未覆盖的 rollout 状态累积"的解释，而非两个架构相同的变体之间存在容量差异。

## 6. 图表清单

- Fig.1 off-policy 失配示意：上为监督状态 vs rollout 状态，下为 6 视角下 on-policy 恢复的定性对比 (P.1)
- Fig.2 方法总览：(A) 稠密/稀疏视角分别重建并沿同一轨迹渲染配对序列；(B) 教师学生共享冻结骨干 + LoRA，几何不对称；(C) $\mathcal{L}_{traj}$ 对齐速度、$\mathcal{L}_{ret}$ 对齐检索响应 (P.2)
- Fig.3 DL3DV 定性对比（GSFixer vs Ours vs GT，反光与欠观测区域）(P.5)
- Fig.4 跨域定性对比（Mip-NeRF 360 与 NeRF-Busters，3/6/9 视角）(P.5-6)
- Fig.5 文本区域定性对比：特权几何恢复粗结构、on-policy 监督修正细笔画 (P.6)
- Fig.6 沿去噪轨迹的逐步质量（PSNR/LPIPS）：后期 on-policy 持续改善、off-policy 平台化 (P.7)
- Table 1 DL3DV 定量对比（3/6/9 视角，5 方法）(P.5)
- Table 2 跨域定量对比（Mip-NeRF 360 与 NeRF-Busters，8 方法）(P.6)
- Table 3 特权几何 + on-policy 监督消融（A 基线 / B-D 教师视角 / E off-policy / F⁻ 无检索对齐 / F 完整）(P.7)
- Algorithm 1 On-policy 蒸馏与部署流程 (P.4)

## 7. 快速总结

**核心贡献**
1. 首次为稀疏视角 3DGS 恢复提出 on-policy 蒸馏：监督状态 = 学生真实访问的 rollout 状态，解决 off-policy 失配与偏差累积。
2. 提出"几何不对称 LUPI"：教师与学生架构/容量相同，唯一差别是稠密 vs 稀疏几何条件；稠密几何只存在于训练期，部署零额外开销。
3. 统一监督机制：特权几何定义"目标是什么"，学生 rollout 定义"在哪儿监督"；教师被查询但从不自己 rollout。
4. 跨域泛化扎实：DL3DV 训练，Mip-NeRF 360 / NeRF-Busters 零适配测试仍全面领先，且稀疏度越低优势越大。

**主要局限**
1. 只针对静态场景（与同类恢复工作一致），动态场景是明确的未来工作。
2. 需要训练场景具备稠密视角（特权信息的前提）；纯稀疏数据源无法使用。
3. 训练代价不低：rank-32 LoRA、每迭代 10 步 rollout 且每步双路径前向（学生 + 冻结教师）。

**可借鉴/可延伸**
1. "目标由特权信息定义、位置由学生 rollout 决定"这个范式可迁移到其他条件扩散蒸馏场景（图像修复、超分、视频补全）。
2. 消融设计很干净：(E)/(F) 对只差"查询哪些状态"，是隔离 exposure bias 效应的好模板。
3. 检索响应对齐（注意力图级蒸馏）比单纯输出对齐多了约束，思路可复用到多视角一致性问题。

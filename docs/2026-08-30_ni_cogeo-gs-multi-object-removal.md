---
arXiv分类: "cs.CV"
领域: "3D 重建 / 神经渲染（3D Gaussian Splatting 场景编辑）"
发表时间: "2026-08-27 (arXiv v1)"
读文章时间: "2026-08-30"
标题: "CoGeo-GS: Concept-Driven and Geometry-Aware Multi-Object Removal in 3D Scenes"
链接: "https://arxiv.org/abs/2608.26656"
Zotero: "已入库"
页码说明: "页码为 PDF 印刷页码（IEEE 格式，首页即 P.1）"
---

# CoGeo-GS: Concept-Driven and Geometry-Aware Multi-Object Removal in 3D Scenes

## 1. TL;DR

针对 3D 场景**多物体移除**任务（遮挡严重、语义纠缠、移除后几何空洞难以补全），本文提出 CoGeo-GS：先用概念感知标记（concept-aware tagging）给每个 3D 高斯分配物体身份，把"选哪些物体删"从逐物体重复优化变成**单次优化内的任意子集选择**；再用"单目深度先验对齐 + 扩散深度补全 + 边界融合 + 几何正则化精修"的管线恢复被遮挡背景的几何，保证多视角一致。在 Mip-NeRF 360 与 SPIn-NeRF 上全面超越 InFusion / GaussianEditor / Gaussian Grouping / SPIn-NeRF 等基线。

## 2. 作者与单位

- 倪远翔 Yuanxiang Ni（南方科技大学 SUSTech）
- 黄宪亮 Xianliang Huang（字节跳动 PICO，通讯作者）
- 马晨航 Chenhang Ma（浙江大学）
- 肖晨 Chen Xiao（复旦大学）
- 马跃文 Yuewen Ma（字节跳动 PICO）
- 王如心 Ruxin Wang（中科院深圳先进院 SIAT，共同通讯）
- 张昊 Hao Zhang（中科院深圳先进院 SIAT，共同通讯）

基金：国家自然科学基金 62472415。

## 3. 紧密相关的工作与交叉点

- **单物体 3D 编辑**：GaussianEditor [6]、InFusion [20]、Gaussian Grouping [7] 在单物体移除上效果好，但多物体时要么逐物体重复优化（算力爆炸、误差累积），要么前后景在表征里纠缠、删一个带偏另一个。
- **2D 扩散修复**：RePaint、LaMa、SmartBrush 等 2D 修复能力强但无 3D 一致性；Inpaint3D 用扩散先验驱动 NeRF，但 NeRF 体渲染架构受限。
- **深度先验**：Depth Anything 3（单目深度）、InFusion（深度扩散补全）。
- **交叉点**：把"3D 语义分割（Gaussian Grouping 的蒸馏思路）+ 深度引导的几何补全（InFusion 的扩散深度补全）+ 精修约束（trust region）"三者合到一个管线，主打**多物体、单阶段、几何稳定**。与同组前作 Semantic-guided progressive object removal [4]（2607.04144）是渐进式单物体思路的延续。

## 4. 问题与核心方法

### 4.1 要解决的问题

给定已重建的 3DGS 场景 $\mathcal{G}$ 和一组概念掩码（要移除的物体），修改 $\mathcal{G}$ 使结果几何完整、视觉一致。三个耦合难点：

1. 可靠地识别并删除目标物体相关的全部高斯；
2. 补全删除区域缺失的几何且不破坏周围背景结构；
3. 所有视角渲染外观一致。

### 4.2 核心方法

整体分四块（Fig.1, P.3）：概念感知标记 → 深度引导几何补全 → 高斯融合 → 几何正则化精修。

**① 概念感知标记（Concept-Aware Tagging）**

3DGS 中每个高斯 $g_i = (\mu_i, \Sigma_i, \alpha_i, \mathbf{c}_i)$（中心、协方差、不透明度、SH 颜色系数）。渲染时颜色为深度排序后的 $\alpha$-blending：

$$
C=\sum_{i\in\mathcal{N}}\mathbf{c}_{i}\alpha_{i}T_{i},\quad T_{i}=\prod_{j=1}^{i-1}(1-\alpha_{j}),
$$

其中 $\mathcal{N}$ 是覆盖该像素的高斯集合，$T_i$ 是累计透射率。

身份标记分两步：

- **2D 标签生成**：对每个目标物体给文本 prompt，用 Grounding DINO 生成文本条件框，SAM 提取各视角掩码；同一组 prompt 与索引顺序在所有视角共享，得到跨视角一致的 2D 标签图 $O(u) \in \{0,1,\ldots,Q\}$（0=背景，$q$=第 $q$ 个目标）。
- **蒸馏到 3D**：给每个高斯学一个可学习特征 $f_i \in \mathbb{R}^D$（$D=16$），渲染时对特征做同样的 $\alpha$-合成得到逐像素特征图：

$$
F(u)=\sum_{i\in\mathcal{N}_{u}}\alpha_{i}(u)T_{i}(u)f_{i},\quad T_{i}(u)=\prod_{j<i}(1-\alpha_{j}(u)),
$$

再用线性分类器 $\Phi:\mathbb{R}^D\to\mathbb{R}^{Q+1}$ + softmax 得到逐像素身份预测：

$$
\bar{O}(u)=\mathrm{softmax}\big(\Phi(F(u))\big).
$$

损失：2D 多类交叉熵 + 3D 空间一致性正则：

$$
\mathcal{L}_{\mathrm{obj}}=-\frac{1}{|P|}\sum_{u\in P}\mathbb{E}_{c\sim O(u)}\left[\log\bar{O}_{c}(u)\right],
$$

$$
p_{i}=\mathrm{softmax}\big(\Phi(f_{i})\big),\qquad
\mathcal{L}_{\mathrm{space}}=\frac{1}{|\Omega|}\sum_{i\in\Omega}\frac{1}{k}\sum_{j\in\mathcal{K}(x_{i})}\mathrm{KL}\big(p_{i}\,\|\,p_{j}\big),
$$

$$
\mathcal{L}_{\mathrm{Dis}}=\mathcal{L}_{\mathrm{obj}}+\lambda\,\mathcal{L}_{\mathrm{space}},
$$

其中 $x_i$ 为高斯 $i$ 的中心，$\mathcal{K}(x_i)$ 是其 $k$ 近邻（$k=5$），$\Omega$ 是采样子集，$\lambda=0.1$。效果：每个高斯获得跨视角稳定的物体身份，遮挡边界附近的高斯也能靠 3D 近邻约束得到监督。

**② 深度引导的几何补全（两阶段）**

删除目标高斯后产生几何空洞。直接 2D 修复 + 反投影会破坏跨视角一致性；纯扩散深度补全会尺度漂移。

- **阶段 1 尺度锚定**：用 Depth Anything 3 估计单目相对深度 $D_{\mathrm{mono}}^{(v)}$，在可靠背景区域 $\Omega_{\mathrm{bg}}^{(v)}$ 上对渲染深度 $D_{\mathrm{render}}^{(v)}$ 做最小二乘拟合尺度/偏移：

$$
(a^{*},b^{*})=\arg\min_{a,b}\sum_{u\in\Omega_{\mathrm{bg}}^{(v)}}\left(a\,D_{\mathrm{mono}}^{(v)}(u)+b-D_{\mathrm{render}}^{(v)}(u)\right)^{2},
$$

得到对齐深度 $D_{\mathrm{align}}^{(v)}=a\cdot D_{\mathrm{mono}}^{(v)}+b$；再用软膨胀掩码 $M_{\mathrm{soft}}^{(v)}$ 与渲染深度混合保证边界平滑：

$$
D_{\mathrm{anchor}}^{(v)}=M_{\mathrm{soft}}^{(v)}\odot D_{\mathrm{align}}^{(v)}+\left(1-M_{\mathrm{soft}}^{(v)}\right)\odot D_{\mathrm{render}}^{(v)}.
$$

- **阶段 2 扩散细节精修**：用预训练深度扩散模型 InFusion 做即插即用先验，输入是修复 RGB、锚定深度、空洞掩码、掩码深度四路潜变量的拼接：

$$
z_{\mathrm{in}}=\mathrm{concat}\big(z_{\mathrm{rgb}},\;z_{d,t},\;z_{\mathrm{mask}},\;z_{\mathrm{mask\_d}}\big),
$$

并在每步采样中做**尺度锚定的掩码约束**——把背景区域深度强制固定为加噪后的锚定深度：

$$
z_{d,t}\leftarrow\left(1-z_{\mathrm{mask}}\right)\odot z_{\mathrm{anchor\_noised},t}+z_{\mathrm{mask}}\odot z_{d,t}.
$$

这样扩散模型只在缺失区域内合成高频几何，全局结构与尺度不乱。反扩散得到补全深度 $D_{\mathrm{inpaint}}^{(v)}$。

**③ 高斯融合（overlap-aware pruning）**

补全深度反投影成点云，转成新高斯集 $\mathcal{G}_{\mathrm{patch}}$。对空洞边界附近的背景高斯 $\mathcal{G}_{\mathrm{bg}}$ 做剪枝，两个条件同时满足才删：

- **近邻重叠**：与新补丁空间相邻，$\min_{k\in\mathcal{G}_{\mathrm{patch}}}\|\mu_i-\mu_k\|<\delta_{\mathrm{near}}$；
- **局部稀疏**：半径 $r$ 内邻居数 $< n_{\min}$（稀疏离群点）。

剩余背景高斯与 $\mathcal{G}_{\mathrm{patch}}$ 合并成初始融合场景 $\mathcal{G}_{\mathrm{init}}$。

**④ 几何正则化精修（两阶段）**

- 先**冻结几何**，只优化 SH 系数与不透明度，用修复图 $I_{\mathrm{Inp}}^{(v)}$ 做参考：

$$
\mathcal{L}_{\mathrm{ref}}=(1-\lambda_{\mathrm{ssim}})\|I_{\mathrm{render}}-I_{\mathrm{Inp}}^{(v)}\|_{1}+\lambda_{\mathrm{ssim}}\left(1-\mathrm{SSIM}(I_{\mathrm{render}},I_{\mathrm{Inp}}^{(v)})\right),
$$

$\lambda_{\mathrm{ssim}}=0.2$，共 150 轮。

- 再放开几何参数修融合接缝，但每个参数张量的更新做**信任域范数裁剪**，防止未编辑区域被带偏：

$$
\Delta\theta\leftarrow\Delta\theta\cdot\min\left(1,\frac{\varepsilon}{\|\Delta\theta\|_{2}}\right).
$$

## 5. 实验

实验设置：Mip-NeRF 360 + SPIn-NeRF 两个基准（有界/无界场景各一）；基线 InFusion、SPIn-NeRF、GaussianEditor、Gaussian Grouping（官方实现重训保证公平）；指标 PSNR / SSIM / LPIPS / FID，**全部只在掩码区域上计算**（严格评估修复质量）；单张 RTX 4090，3DGS 30k 迭代，单目深度 504×504，扩散精修 768×768。

**多物体移除（Task 1, Tab.I, P.5）**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| GaussianEditor | 22.4 | 0.742 | 0.284 | 31.6 |
| InFusion | 24.8 | 0.803 | 0.213 | 24.9 |
| Gaussian Grouping | 25.6 | 0.821 | 0.198 | 22.8 |
| SPIn-NeRF | 24.9 | 0.812 | 0.221 | 23.7 |
| **CoGeo-GS** | **28.9** | **0.882** | **0.086** | **11.4** |

LPIPS 较最强基线降 56.6%、FID 降 50.0%。

**单物体移除（Task 2, Tab.I, P.5）**：PSNR 30.7 / SSIM 0.903 / LPIPS 0.072 / FID 9.8，同样全面领先。

**定性（Fig.2/Fig.3, P.5）**：GaussianEditor/InFusion 多物体时出现黑洞、边界不一致与漂浮伪影（尺度不匹配）；Gaussian Grouping 有漂浮伪影且背景与修复区外观不一致；CoGeo-GS 结构完整、细节清晰。

**消融（Tab.II, P.5 + Fig.4, P.6）**

| 变体 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| w/o 概念感知掩码（退化为 2D 投影删高斯） | 26.85 | 0.851 | 0.112 | 16.90 |
| w/o 深度引导（无尺度锚定的扩散补全） | 25.97 | 0.832 | 0.128 | 18.40 |
| **完整模型** | **28.90** | **0.882** | **0.086** | **11.40** |

- 去掉概念感知掩码 → PSNR 明显下降：2D 投影受视角可见性影响，部分遮挡高斯删不干净，残留伪影（Fig.4）。
- 去掉深度引导 → PSNR/FID 明显掉，出现漂浮伪影与几何错位（Fig.4），无尺度锚定时重建高斯表面几何不稳定。

## 6. 图表清单

- Fig.1 框架总览：概念标记 → 深度补全 → 融合剪枝 → 两阶段精修 (P.3)
- Fig.2 多物体移除定性对比（GaussianEditor / InFusion / Gaussian Grouping / Ours）(P.5)
- Fig.3 单物体移除定性对比（SPIn-NeRF / InFusion / Ours，室内外场景）(P.5)
- Fig.4 消融定性：有无概念标记、有无深度引导的对比 (P.6)
- Table I 多物体（Task 1）与单物体（Task 2）定量对比，4 指标 (P.5)
- Table II 消融定量（w/o 概念掩码 / w/o 深度引导 / 完整）(P.5)

## 7. 快速总结

- 核心贡献：概念感知标记让多物体移除在**单阶段优化**内完成，任意子集可删，前后景解耦；几何补全用"单目深度锚定 + 扩散细化 + 边界融合"保证跨视角一致；信任域约束的精修不破坏未编辑区域。
- 方法亮点：把 2D 语义蒸馏成每高斯 16 维特征 + 3D 近邻 KL 正则，是稳健 3D 选择的低成本方案。
- 局限（论文未强调但可见）：依赖 Grounding DINO + SAM 的 2D 掩码质量（文本 prompt 表达不准确的物体可能漏标）；扩散补全 768×768 的耗时；指标只在掩码区计算，未报告完整场景渲染质量。
- 可延伸：物体身份特征 $f_i$ 天然可复用于"选哪些删/留/换"的统一编辑；信任域精修思路可推广到其他 3D 编辑后处理。
- 与同组前作 [4] 2607.04144（语义引导渐进式移除）构成连续工作线，值得一起读。

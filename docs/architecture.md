# Architecture & Methodology

## 1. System Overview

Ultra-Wideband (UWB) radio signals provide centimeter-level ranging capabilities in Line-of-Sight (LOS) conditions. However, in complex indoor environments, Non-Line-of-Sight (NLOS) propagation—caused by walls, obstacles, and multipath reflections—severely degrades positioning accuracy by introducing positive ranging bias and channel distortion.

**UWB-MoE** addresses this challenge via a deep Mixture-of-Experts (MoE) architecture that dynamically identifies channel degradation per anchor, routes representations through regime-specialized experts (LOS vs. NLOS), and fuses spatial observations using attention pooling.

```
                      +-----------------------------+
                      |   Anchor CIR Envelopes      |
                      |   + Time Differences (TD)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       CIREncoder (1D CNN)   |
                      |  Conv1d -> ReLU -> Conv1d   |
                      |  -> AdaptiveAvgPool1d       |
                      |  -> Concat with TD & Emb    |
                      +--------------+--------------+
                                     |
                         +-----------+-----------+
                         |                       |
                         v                       v
               +-------------------+   +-------------------+
               |     SoftGate      |   |   Condition       |
               | (Routing Network) |   |   Experts         |
               +---------+---------+   |  - LOS Expert     |
                         |             |  - NLOS Expert    |
                         | (Weights)   +---------+---------+
                         |                       |
                         +-----------+-----------+
                                     |
                                     v
                      +-----------------------------+
                      |     Soft Mixture Fusion     |
                      | z = w_LOS * e_LOS           |
                      |     + w_NLOS * e_NLOS       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Multi-Anchor Attention    |
                      |   - Dynamic Masking         |
                      |   - Attention Pooling       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Position Regression      |
                      |    Head: (X, Y) Coordinates|
                      +-----------------------------+
```

---

## 2. Mathematical Formulation

### 2.1 Multi-Modal CIR Encoding
Given an anchor measurement with Channel Impulse Response real and imaginary components $\mathbf{h} = \mathbf{h}_r + j \mathbf{h}_i$, the log magnitude envelope is computed as:
$$\mathbf{m}[k] = \log\left(1 + \sqrt{\mathbf{h}_r[k]^2 + \mathbf{h}_i[k]^2}\right)$$

The envelope $\mathbf{m}$ is processed by a 2-stage 1D convolution block followed by adaptive average pooling:
$$\mathbf{f}_{\text{cir}} = \text{AdaptiveAvgPool1d}(\text{ReLU}(\text{Conv1d}(\text{ReLU}(\text{Conv1d}(\mathbf{m})))))$$

The temporal feature $\mathbf{f}_{\text{cir}}$ is concatenated with normalized Time Difference ($\tau$), TD offset ($\Delta \tau$), and the anchor's learned spatial embedding $\mathbf{e}_a$:
$$\mathbf{x}_{\text{fused}} = \text{Linear}\left([\mathbf{f}_{\text{cir}} \,\|\, \tau \,\|\, \Delta \tau \,\|\, \mathbf{e}_a]\right)$$

### 2.2 Soft Mixture-of-Experts (MoE) Routing
The soft gating network computes a categorical distribution over $K=2$ experts (LOS and NLOS):
$$\mathbf{w} = \text{Softmax}(\mathbf{W}_g \mathbf{x}_{\text{fused}} + \mathbf{b}_g)$$

The mixed representation $\mathbf{z}$ combines the outputs of the Line-of-Sight expert $E_{\text{los}}$ and Non-Line-of-Sight expert $E_{\text{nlos}}$:
$$\mathbf{z} = w_{\text{los}} E_{\text{los}}(\mathbf{x}_{\text{fused}}) + w_{\text{nlos}} E_{\text{nlos}}(\mathbf{x}_{\text{fused}})$$

### 2.3 Load Balancing & Diversity Regularization
To prevent **expert collapse** (where the router routes all samples to a single expert and starves the other), we apply a load-balancing diversity loss:
$$\mathcal{L}_{\text{diversity}} = K \sum_{i=1}^K \bar{w}_i^2 - \left(\sum_{i=1}^K \bar{w}_i\right)^2$$
where $\bar{w}_i = \frac{1}{B} \sum_{b=1}^B w_{b, i}$ is the batch-averaged gating probability for expert $i$.

- When expert assignments are uniform ($\bar{w}_i = \frac{1}{K}$), $\mathcal{L}_{\text{diversity}} = 0$.
- When collapsed to one expert ($\bar{w}_1 = 1, \bar{w}_2 = 0$), $\mathcal{L}_{\text{diversity}} = K - 1 > 0$.

### 2.4 Active Spatial Anchor Attention
For a burst containing $A$ anchors, let $\mathbf{z}_a \in \mathbb{R}^{D}$ be the feature representation for anchor $a$. Anchors with zeroed telemetry are masked out:
$$s_a = \mathbf{v}^\top \mathbf{z}_a, \quad \tilde{s}_a = \begin{cases} s_a & \text{if anchor } a \text{ is active} \\ -\infty & \text{if anchor } a \text{ is masked} \end{cases}$$

$$\alpha_a = \frac{\exp(\tilde{s}_a)}{\sum_{j=1}^A \exp(\tilde{s}_j)}$$

The fused global embedding is computed via convex attention pooling:
$$\mathbf{z}_{\text{global}} = \sum_{a=1}^A \alpha_a \mathbf{z}_a$$

### 2.5 Coordinate Regression & Total Loss
The coordinate regression head predicts Cartesian coordinates:
$$\hat{\mathbf{p}} = [\hat{x}, \hat{y}] = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{z}_{\text{global}} + \mathbf{b}_1) + \mathbf{b}_2$$

The model is trained end-to-end minimizing the composite objective:
$$\mathcal{L} = \|\hat{\mathbf{p}} - \mathbf{p}^*\|_2^2 + \lambda \mathcal{L}_{\text{diversity}}$$
where $\lambda$ (default: `0.01`) balances accuracy with gating diversity.

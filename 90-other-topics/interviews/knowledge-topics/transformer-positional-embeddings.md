# Positional Encodings in Transformers: Sinusoidal PE vs. RoPE

This document summarizes the mathematical structure, high-dimensional geometry, and representation-learning mechanisms of **Additive Sinusoidal Positional Embeddings** and **Rotary Position Embeddings (RoPE)**.

---

## 1. Sinusoidal Formulation and Relative Position Invariance

### Mathematical Definition
Self-attention is permutation-invariant by default. To inject token order without learned parameters or a hard sequence-length cap, sinusoidal positional embeddings assign each position $\text{pos}$ and dimension index $i$ (for model dimension $d_{\text{model}}$) a deterministic wave:

$$\text{PE}(\text{pos}, 2i) = \sin(\omega_i \cdot \text{pos}), \qquad \text{PE}(\text{pos}, 2i + 1) = \cos(\omega_i \cdot \text{pos})$$

where the angular frequency $\omega_i$ decreases geometrically across dimension pairs:

$$\omega_i = \frac{1}{10000^{2i / d_{\text{model}}}}$$

* **Even dimensions ($2i$):** Encoded with sine functions.
* **Odd dimensions ($2i + 1$):** Encoded with cosine functions.
* **Multi-scale wavelengths:** Smaller $i$ produces rapid oscillations for fine local position tracking; larger $i$ produces slow oscillations for long-range context.

### Relative Position via Linear Transformations
Using trigonometric angle-addition identities, the positional encoding at $\text{pos} + k$ can be written as a linear transformation of the encoding at $\text{pos}$:

$$\begin{bmatrix} \text{PE}(\text{pos} + k, 2i) \\ \text{PE}(\text{pos} + k, 2i + 1) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \text{PE}(\text{pos}, 2i) \\ \text{PE}(\text{pos}, 2i + 1) \end{bmatrix}$$

Across all $d_{\text{model}} / 2$ pairs, this forms a block-diagonal orthogonal rotation matrix $M_k$ that depends **only on the relative offset $k$** and not on the absolute position $\text{pos}$:

$$\text{PE}(\text{pos} + k) = M_k \, \text{PE}(\text{pos})$$

* **Why position invariance matters:** Because $M_k$ is translation-equivariant, a set of learned attention weights can learn a relative syntactic rule (such as attending $k = -1$ token back) once and apply it consistently anywhere in a sequence, including on unseen sequence lengths.

---

## 2. Why Positional Encodings Occupy a Low-Dimensional Subspace

Although $\text{PE}(\text{pos}) \in \mathbb{R}^{d_{\text{model}}}$, the vectors trace a smooth 1D curve parameterized by a single scalar $\text{pos}$. Over any finite context length $L$, positional vectors occupy a low-rank subspace—a property explained by two equivalent mathematical views:

### 1. Geometric / PCA View (Singular Value Decay)
For large dimension indices $i$, $\omega_i \cdot L \ll 1$, so $\sin(\omega_i \cdot \text{pos}) \approx \omega_i \cdot \text{pos}$ and $\cos(\omega_i \cdot \text{pos}) \approx 1$. Hundreds of slow-frequency columns behave nearly identically across the sequence.
* **Empirical rank ($d_{\text{model}} = 512, L = 1024$):** Performing Singular Value Decomposition (SVD) on the centered positional encoding matrix $P \in \mathbb{R}^{1024 \times 512}$ shows that **155 principal components (30.3%) capture 99.99% of positional variance**, with singular values plunging to zero immediately past index ~160.
* **Orthogonal headroom:** Roughly 70% of $\mathbb{R}^{d_{\text{model}}}$ experiences virtually zero positional movement.

### 2. Algebraic / Linear Dependency View (Finite Horizon)
Because the low-frequency rotation blocks in $M_1$ are near-identity matrices, repeatedly stepping forward along a finite sequence keeps those coordinates trapped on shared low-degree polynomials ($1, \text{pos}, \text{pos}^2$). Consequently, any position vector in that slow regime can be written as a direct scalar linear combination of a small basis of position vectors, restricting the span to the exact same low-rank subspace.

---

## 3. Dot-Product Cross-Terms: Additive PE vs. RoPE

### The Four Terms in Additive Sinusoidal PE
When position is added to token embeddings before the attention projections ($\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i$), expanding the query-key dot product $\mathbf{q}_i^\top \mathbf{k}_j = (W_Q \mathbf{x}_i)^\top (W_K \mathbf{x}_j)$ yields four additive components:

$$\mathbf{q}_i^\top \mathbf{k}_j = \underbrace{\mathbf{e}_i^\top W_Q^\top W_K \mathbf{e}_j}_{\text{1. Content–Content}} + \underbrace{\mathbf{p}_i^\top W_Q^\top W_K \mathbf{p}_j}_{\text{2. Position–Position}} + \underbrace{\mathbf{e}_i^\top W_Q^\top W_K \mathbf{p}_j}_{\text{3. Content–Position}} + \underbrace{\mathbf{p}_i^\top W_Q^\top W_K \mathbf{e}_j}_{\text{4. Position–Content}}$$

* **Interference from cross-terms:** Terms 3 and 4 pair word identity with absolute position vectors, introducing spurious word-position biases that break strict translation invariance and degrade length extrapolation unless suppressed during training.

### How RoPE Eliminates Cross-Terms
**Rotary Position Embeddings (RoPE)** avoid adding $\mathbf{p}_i$ to the input embeddings. Instead, RoPE projects pure content states first and applies the block-diagonal rotation matrix $M_i$ directly to queries and keys:

$$\tilde{\mathbf{q}}_i = M_i W_Q \mathbf{e}_i, \qquad \tilde{\mathbf{k}}_j = M_j W_K \mathbf{e}_j$$

Because $M_i$ is orthogonal ($M_i^\top M_j = M_{j - i}$), the inner product collapses into a single relative term:

$$\tilde{\mathbf{q}}_i^\top \tilde{\mathbf{k}}_j = \mathbf{e}_i^\top W_Q^\top M_{j - i} W_K \mathbf{e}_j = \sum_{m=0}^{d_{\text{head}}/2 - 1} r_q^{(m)} r_k^{(m)} \cos\Big(\phi_k^{(m)} - \phi_q^{(m)} + (j - i)\omega_m\Big)$$

---

## 4. How Models Separate Positional and Semantic Information: Additive PE vs. RoPE

Both architectures allow Transformers to disentangle position from token semantics, but they rely on fundamentally different geometric and spectral mechanisms.

### Mechanism Comparison

| Dimension | Additive Sinusoidal PE | Rotary Position Embeddings (RoPE) |
| :--- | :--- | :--- |
| **Primary Separation Axis** | **Orthogonal geometric subspaces** in $\mathbb{R}^{d_{\text{model}}}$ | **Frequency bands** (high-frequency vs. low-frequency 2D pairs) |
| **Semantic Head Behavior** | Places the low-rank positional subspace in the **null space** of $W_Q, W_K$ ($W_Q \mathbf{p}_i \approx \mathbf{0}$) | Routes energy into **low-frequency** pairs where $(j - i)\omega_m \approx 0$ (identity rotation) |
| **Positional Head Behavior** | Suppresses token variance ($W_Q \mathbf{e}_i \approx \mathbf{0}$) and reads directly from the positional subspace | Maps all tokens onto a **fixed 2D reference vector** in **high-frequency** pairs with a pre-rotated phase offset $-k\omega_m$ |
| **Hybrid Head Behavior** | Projects onto both subspaces simultaneously, managing cross-term noise | Splits energy across low frequencies (semantic match) and mid/high frequencies (locality decay) |
| **Residual Stream & $V$ Vectors** | Mixed with static positional waves from layer 1 onward | **Completely unpolluted**; rotation is applied only inside $Q$ and $K$ |

### Detailed Breakdown of Each Strategy

#### A. Additive Sinusoidal PE: Subspace Orthogonality & Null-Space Filtering
1. **Geometric partitioning:** Because $\mathbf{PE}(\text{pos})$ occupies only a ~30% low-rank subspace, gradient descent shapes the token embedding matrix $E$ to store semantics primarily in the orthogonal complement.
2. **Projection filtering:** By linearity, $W_Q(\mathbf{e}_i + \mathbf{p}_i) = W_Q \mathbf{e}_i + W_Q \mathbf{p}_i$. Pure content heads learn weights whose null space covers the dominant positional directions ($W_Q \mathbf{p}_i \approx \mathbf{0}$), while structural heads (like previous-token heads) suppress token variance ($W_Q \mathbf{e}_i \approx \mathbf{0}$).
3. **Multi-layer residual routing:** Early positional heads copy neighboring value vectors into the residual stream so downstream semantic heads can perform content matching across context without relying on raw positional coordinates.

#### B. RoPE: Frequency-Band Specialization & Constant-Vector Projections
1. **Low-frequency semantic channels:** For large pair indices $m$, $\omega_m$ is tiny. Even over hundreds of tokens, $(j - i)\omega_m \approx 0$, so $R_{(j-i)\omega_m} \approx I$. Semantic heads concentrate weight norms in these slow channels to achieve distance-invariant content matching.
2. **High-frequency positional circuits:** In specialized positional heads (e.g., targeting offset $k = -1$), $W_Q^{(m)}$ and $W_K^{(m)}$ place token-specific semantic directions in their null space and map **every token** onto the same fixed 2D reference vector $\mathbf{v}_{\text{ref}}$ within the fast-rotating bands ($m = 0, 1, 2, \dots$), baking in a relative phase offset $-k\omega_m$.
   * When $j - i = k$, the rotation cancels the phase offset ($(j - i - k)\omega_m = 0$), causing all high-frequency cosines to peak simultaneously (**constructive interference**).
   * At any other offset $j - i \neq k$, the fast waves oscillate out of phase and cancel out (**destructive interference**).
3. **Clean value propagation:** Because RoPE never rotates $W_V \mathbf{h}_j$ or adds vectors to the residual stream, positional geometry controls attention routing without ever consuming embedding dimensions or polluting semantic features.

---

## References & Further Reading

* **Foundational Architecture:** Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — Introduces sinusoidal positional encodings and notes their linear relative-position property.
* **Rotary Position Embeddings:** Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) — Derives RoPE from relative inner-product invariance and analyzes long-term decay.
* **RoPE Frequency Specialization:** Barbero et al. (2024), [*Round and Round We Go! What Makes Rotary Position Embeddings Useful?*](https://arxiv.org/abs/2410.06205) — Demonstrates how trained Transformers use high-frequency bands for positional heads and low-frequency bands for semantic attention.
* **Subspace Disentanglement & Interpretability:**
  * Lequeu et al. (2026), [*Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders*](https://arxiv.org/abs/2605.30022) — Examines the low-dimensional positional manifold, subspace orthogonality, and semantic vs. structural head specialization.
  * Harvard Data Science Review (2026), [*Uncovering Hidden Geometry in Transformers*](https://hdsr.mitpress.mit.edu/pub/l9iaw8m3/download/pdf) — Analyzes $QK$ circuit decompositions across content and positional cross-terms.
* **Implementation & Conceptual Guides:**
  * [LearnOpenCV: Sinusoidal Position Embeddings](https://learnopencv.com/sinusoidal-position-embeddings/)
  * [GeeksforGeeks: Positional Encoding in Transformers](https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/)
  * [Cross Validated: Intuitive Explanation for Summing Embedding and Positional Encoding](https://stats.stackexchange.com/questions/586813/intuitive-explanation-for-summing-the-embedding-and-positional-encoding-in-the-t)

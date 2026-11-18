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
	1. **$W_{Q}^{(m)}$ refers to the $2 \times d_k$ slice (or block of rows) of $W_{Q}$ that acts specifically on the m-th 2D frequency subspace** (dimensions 2m and 2m+1 of the input).
	2. See [[#^2bfe07]]
	3. When $j - i = k$, the rotation cancels the phase offset ($(j - i - k)\omega_m = 0$), causing all high-frequency cosines to peak simultaneously (**constructive interference**).
   * At any other offset $j - i \neq k$, the fast waves oscillate out of phase and cancel out (**destructive interference**).
3. **Clean value propagation:** Because RoPE never rotates $W_V \mathbf{h}_j$ or adds vectors to the residual stream, positional geometry controls attention routing without ever consuming embedding dimensions or polluting semantic features.

---

## References & Further Reading

* **Additive Sinusoidal PE:** ExplainingAI, [*Positional Encoding in Transformer | Sinusoidal Positional Encoding Explained*](https://www.youtube.com/watch?v=dWkm4nFikgM) — Explains why such a design from 1st principle.
* **Rotary Position Embeddings:** Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) — Derives RoPE from relative inner-product invariance and analyzes long-term decay.
* **RoPE Frequency Specialization:** Barbero et al. (2024), [*Round and Round We Go! What Makes Rotary Position Embeddings Useful?*](https://arxiv.org/abs/2410.06205) — Demonstrates how trained Transformers use high-frequency bands for positional heads and low-frequency bands for semantic attention.
* **Derive RoPE via Group theory:** — Alok Puranik [*Using group theory to explore the space of positional encodings for attention*](https://blog.janestreet.com/using-group-theory-to-explore-positional-encodings-attention/). There is an associated video as well.
* **Subspace Disentanglement & Interpretability:**
  * Lequeu et al. (2026), [*Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders*](https://arxiv.org/abs/2605.30022) — Examines the low-dimensional positional manifold, subspace orthogonality, and semantic vs. structural head specialization.
  * Harvard Data Science Review (2026), [*Uncovering Hidden Geometry in Transformers*](https://hdsr.mitpress.mit.edu/pub/l9iaw8m3/download/pdf) — Analyzes $QK$ circuit decompositions across content and positional cross-terms.
* **Implementation & Conceptual Guides:**
  * [LearnOpenCV: Sinusoidal Position Embeddings](https://learnopencv.com/sinusoidal-position-embeddings/)
  * [GeeksforGeeks: Positional Encoding in Transformers](https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/)
  * [Cross Validated: Intuitive Explanation for Summing Embedding and Positional Encoding](https://stats.stackexchange.com/questions/586813/intuitive-explanation-for-summing-the-embedding-and-positional-encoding-in-the-t)

---
# High-Frequency Positional Circuits in Transformers

^2bfe07

This document summarizes how Transformer attention heads isolate relative positional offsets (such as $k = -1$) while ignoring token semantics, using subspace orthogonalization, low-rank projections, and sinusoidal frequency bands.

---

## 1. Core Mechanism: What $W_Q^{(m)}$ Means

In sinusoidal positional encoding, a model embedding of dimension $d_{\text{model}}$ is partitioned into $d_{\text{model}}/2$ pairs of dimensions. Each pair $m \in \{0, 1, \dots, d_{\text{model}}/2 - 1\}$ corresponds to a specific sinusoidal angular frequency $\omega_m$:

$$\text{PE}_{(\text{pos}, 2m)} = \sin(\text{pos} \cdot \omega_m), \qquad \text{PE}_{(\text{pos}, 2m+1)} = \cos(\text{pos} \cdot \omega_m)$$

For a given attention head, the query weight matrix $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ maps the full input vector $\mathbf{x}_i = \mathbf{e}_{w_i} + \mathbf{p}_i$ into the head's query space. 

* **$W_Q^{(m)}$** denotes the $2 \times d_k$ slice (block of rows) of $W_Q$ that acts exclusively on the $m$-th 2D frequency subspace (dimensions $2m$ and $2m+1$).
* The full query vector can be expressed as a sum over all frequency bands:

$$\mathbf{q}_i = \mathbf{x}_i W_Q = \sum_{m=0}^{d_{\text{model}}/2 - 1} \mathbf{x}_i^{(m)} W_Q^{(m)}$$

where $\mathbf{x}_i^{(m)} \in \mathbb{R}^2$ is the 2D slice of the input embedding at frequency band $m$.

---

## 2. Terminology: Frequency Bands and Small Rank

### Frequency Band
* **Definition:** A single sine-cosine pair oscillating at a fixed rate $\omega_m$. Because $\sin(\text{pos}\cdot\omega_m)$ and $\cos(\text{pos}\cdot\omega_m)$ trace a unit circle, each band represents a 2D rotating vector $\mathbf{p}_i^{(m)}$.
* **High vs. Low Bands:** Low indices ($m = 0, 1, 2, \dots$) rotate rapidly and change noticeably from one token to the next (useful for short-range relative offsets). Higher indices rotate slowly over long distances.

### Small Rank
* **Definition:** The **rank** of a weight matrix $W_Q$ is the number of independent input dimensions it actively reads. All remaining dimensions fall into its **null space** and are mapped to $\mathbf{0}$.
* **Application to Positional Heads:** Reading a single 2D frequency band requires a rank of $2$. If a positional head only reads $5$ fast-rotating bands ($m = 0, 1, 2, 3, 4$) to detect an adjacent token, $W_Q$ only requires a rank of $5 \times 2 = 10$.
* **Result:** In a model with $d_{\text{model}} = 512$, a rank-$10$ matrix leaves $512 - 10 = 502$ dimensions in its null space.

---

## 3. How All Token Semantics Fit in the Null Space

A single linear projection can zero out thousands of distinct token meanings simultaneously because **the null space only needs to cover the subspace where semantics live, not the entire vocabulary size.**

1. **Subspace Orthogonalization:** Because token embeddings $\mathbf{e}_w$ are learned while sinusoidal encodings $\mathbf{p}_i$ are fixed, gradient descent separates them into orthogonal subspaces to prevent interference. Learned token embeddings place their semantic variance into dimensions outside the high-frequency positional channels ($\mathbf{e}_w^{(m)} \approx \mathbf{0}$).
2. **Rank-Nullity Filtering:** Because $\mathbf{e}_w$ resides inside the $\sim 500$-dimensional null space of $W_Q$ and $W_K$, multiplying $\mathbf{e}_w W_Q$ produces $\mathbf{0}$ for every token in the vocabulary:

$$\mathbf{q}_i = (\mathbf{e}_{w_i} + \mathbf{p}_i) W_Q = \underbrace{\mathbf{e}_{w_i} W_Q}_{\mathbf{0}} + \mathbf{p}_i W_Q = \mathbf{p}_i W_Q$$

---

## 4. Concrete Matrix Example ($k = -1$)

Consider a minimal $4\text{D}$ model space ($d_{\text{model}} = 4$) and a single attention head ($d_k = 2$):
* **Dimensions $0, 1$ (Band $m=0$, Positional):** Holds frequency $\omega_0$.
* **Dimensions $2, 3$ (Semantic Subspace):** Holds token-specific semantic features $(s_{w,1}, s_{w,2})$.

For token $w$ at position $i$:

$$\mathbf{x}_i = \mathbf{e}_w + \mathbf{p}_i = \begin{bmatrix} \sin(i \omega_0) & \cos(i \omega_0) & s_{w,1} & s_{w,2} \end{bmatrix}$$

### Weight Matrix Construction
To attend strictly to the preceding token ($k = -1$), we zero out the semantic rows and apply a $-k\omega_0 = +\omega_0$ rotation in $W_Q^{(0)}$ relative to $W_K^{(0)}$ (scaled by $c$ to sharpen softmax):

$$W_Q = c \begin{bmatrix} \cos(\omega_0) & -\sin(\omega_0) \\ \sin(\omega_0) & \cos(\omega_0) \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \qquad W_K = c \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$$

### Projections and Dot Product
1. **Key at position $j$:**
   $$\mathbf{k}_j = \mathbf{x}_j W_K = c \begin{bmatrix} \sin(j\omega_0) & \cos(j\omega_0) \end{bmatrix}$$

2. **Query at position $i$:** Using angle addition identities, the rotation shifts the phase backward by one step:
   $$\mathbf{q}_i = \mathbf{x}_i W_Q = c \begin{bmatrix} \sin((i - 1)\omega_0) & \cos((i - 1)\omega_0) \end{bmatrix}$$

3. **Attention Score ($\mathbf{q}_i \cdot \mathbf{k}_j$):**
   $$\begin{aligned}
   \mathbf{q}_i \cdot \mathbf{k}_j &= c^2 \Big[ \sin((i - 1)\omega_0)\sin(j\omega_0) + \cos((i - 1)\omega_0)\cos(j\omega_0) \Big] \\
   &= c^2 \cos\big( (j - i + 1)\omega_0 \big)
   \end{aligned}$$

* **Constructive Interference ($j - i = -1$):** The phase difference becomes $(-1 + 1)\omega_0 = 0$. Across all active high-frequency bands $m$, $\cos(0) = 1$, causing the dot product to peak simultaneously at its maximum value.
* **Destructive Interference ($j - i \neq -1$):** Across multiple high-frequency bands, the cosines oscillate out of phase and cancel out, suppressing attention to all other positions.

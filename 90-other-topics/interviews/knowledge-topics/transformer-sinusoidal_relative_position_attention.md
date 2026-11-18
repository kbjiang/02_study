# Sinusoidal Positional Encodings & Relative Position Attention
> This doc aims to understand the statement that "It allows the model to easily learn to attend by relative positions, because for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}."

## 1. Linear Shift Property of Sinusoidal Encodings
In *Attention Is All You Need*, each frequency $\omega_i = 10000^{-2i/d_{\text{model}}}$ defines a sine-cosine pair at dimensions $2i$ and $2i+1$:

$$\text{PE}(\text{pos}, 2i) = \sin(\omega_i \cdot \text{pos}), \qquad \text{PE}(\text{pos}, 2i+1) = \cos(\omega_i \cdot \text{pos})$$

Using angle-addition trigonometric identities, shifting a position by a fixed offset $k$ expands to:

$$\begin{bmatrix} \text{PE}(\text{pos}+k, 2i) \\ \text{PE}(\text{pos}+k, 2i+1) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \text{PE}(\text{pos}, 2i) \\ \text{PE}(\text{pos}, 2i+1) \end{bmatrix}$$

Across all $d_{\text{model}}/2$ frequency pairs, this forms a block-diagonal rotation matrix $M_k \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:

$$\text{PE}_{\text{pos}+k} = M_k \, \text{PE}_{\text{pos}}$$

* **Position Independence of $M_k$:** The matrix $M_k$ depends solely on the relative offset $k$ and frequencies $\omega_i$, not on the absolute position $\text{pos}$.
* **Norm Preservation:** Each $2 \times 2$ block is an orthogonal rotation matrix, preserving vector norms across shifts.

---

## 2. Single-Step Linear Alignment ($W_Q = W_K M_k$)
When projecting queries and keys from positional vectors:

* **Query at $\text{pos}$:** $q_{\text{pos}} = W_Q \, \text{PE}_{\text{pos}}$
* **Key at $\text{pos}+k$:** $k_{\text{pos}+k} = W_K \, \text{PE}_{\text{pos}+k} = (W_K M_k) \, \text{PE}_{\text{pos}}$

Because $W_K M_k$ is a single matrix product, a single linear layer $W_Q$ can absorb the shift transformation directly.

* **Why This Matters:**
  * **Efficiency:** No extra layers or non-linearities are needed to compute relative offsets.
  * **Translation Invariance:** If $W_Q = W_K M_k$ within an attention head's subspace, $q_{\text{pos}}$ and $k_{\text{pos}+k}$ align identically at every position $\text{pos}$ simultaneously.
  * **Length Generalization:** The rotation property holds even at sequence lengths beyond those seen during training.
* **Additive PE Limitation (vs. RoPE):** Adding $\text{PE}$ directly to token embeddings ($x + \text{PE}$) introduces token-position cross terms in the dot product. Rotary Position Embeddings (RoPE) avoid this cross-term noise by applying the rotation matrix directly to $q$ and $k$ after projection.

---

## 3. Gradient Descent Dynamics ($W_Q \to W_K M_k$)
When attending from $\text{pos}$ to $\text{pos}+k$ reduces the training loss, backpropagation updates $W_Q$ using the gradient of the pre-softmax logit $s = q_{\text{pos}}^\top k_{\text{pos}+k}$:

$$\frac{\partial s}{\partial W_Q} = k_{\text{pos}+k} \, \text{PE}_{\text{pos}}^\top = (W_K M_k \, \text{PE}_{\text{pos}}) \, \text{PE}_{\text{pos}}^\top$$

Accumulating updates across all positions in a sequence gives:

$$\Delta W_Q \propto \sum_{\text{pos}} (W_K M_k) \left(\text{PE}_{\text{pos}} \, \text{PE}_{\text{pos}}^\top\right) = (W_K M_k) \sum_{\text{pos}} \text{PE}_{\text{pos}} \, \text{PE}_{\text{pos}}^\top$$

### Why $\frac{1}{L} \sum_{\text{pos}=0}^{L-1} \text{PE}_{\text{pos}} \text{PE}_{\text{pos}}^\top \approx 0.5 \, I$
* **Diagonal Entries ($a = b$):** Average values of $\sin^2(\omega_i \cdot \text{pos})$ and $\cos^2(\omega_i \cdot \text{pos})$ converge to $0.5$.
* **Same-Frequency Off-Diagonals ($2i$ vs. $2i+1$):** $\sin(\omega_i \cdot \text{pos})\cos(\omega_i \cdot \text{pos}) = \frac{1}{2}\sin(2\omega_i \cdot \text{pos})$ averages to $0$.
* **Cross-Frequency Off-Diagonals ($i \neq j$):** Product-to-sum identities yield oscillating waves at $\omega_i \pm \omega_j$ that cancel to $0$.
* **Finite-$L$ Caveat:** Ultra-low frequencies whose wavelengths exceed sequence length $L$ deviate slightly from $0.5$ on the bottom-right diagonal entries, while higher frequencies converge tightly to $0.5 \, I$.

---

## 4. Balancing Semantics and Position Across Heads
* **Orthogonal Subspace Partitioning:** Within a single head ($d_k = 64$), distinct rows of $W_Q$ and $W_K$ can specialize in semantic token features ($W_Q^{\text{sem}} \text{PE}_{\text{pos}} \approx 0$) while others specialize in positional shifts ($W_Q^{\text{pos}} x_{\text{pos}} \approx 0$). The resulting dot product adds the semantic score and relative-offset score cleanly before softmax.
* **Multi-Head Frequency Specialization:**
  * **Short-Range Heads:** Weight high-frequency dimensions (small $i$) to resolve fine-grained local offsets (e.g., $k = -1$).
  * **Long-Range Heads:** Zero out high frequencies (which alias over long spans) and rely on low-frequency dimensions (large $i$) to track distant dependencies cleanly.

tags: #UnderstandTransformers #PositionalEmbedding
### Positional embedding
1. Additional positional encoding, as in original Transformers paper
2. https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
	1. it really is use Sinusoids with different frequencies/phases to encode different positions. For e.g., this is how you'd do with binary encoding.![[Pasted image 20240615211543.png]]
	2. "We chose this function (me: Sinusoid) because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $\phi$, $PE_{pos+\phi}$ can be represented as a linear function of $PE_{pos}$."
		1. Matrix $M$ has no dependency on position $t$. $$\begin{align*} 
			&M.\begin{bmatrix} sin(\omega_k.t) \\ cos(\omega_k.t) \end{bmatrix}
						= \begin{bmatrix} sin(\omega_k.(t+\phi)) \\ cos(\omega_k.(t + \phi)) \end{bmatrix} \\
						\\
			&M_{\phi, k} = \begin{bmatrix} cos(\omega_k.\phi) & sin(\omega_k.\phi) \\ 
						-sin(\omega_k.\phi) & cos(\omega_k.\phi) \end{bmatrix} \\
			 \end{align*} $$
		 2. Pair of $\omega_k$ is in embedding dimension, while $t$ and $\phi$ are in position dimension. 
3. [王木头](https://youtu.be/GGLr-TtKguA?t=4096)
	1. The whole $P$ matrix is a Fourier expansion. 
		1. think of the values at one position, it's just the list of Fourier coefficients, one at each dimension.
	2. If we collapse all columns we get $f(t)$ with $t$ as token positions. If expanded, in the $t$ direction, we have simple sinusoids with different frequencies in each column
	3. The large base (10,000) in denominator allows long wavelength at high dimensional (equivalently, lower frequency), which allows larger $t$ without repeating the encoding.
		1. too large of a base may lose accuracy at starting positions?
	4. ![[Pasted image 20240615210146.png|800]]
4. Nice [video](https://youtu.be/T3OT8kqoqjc)with explanation and visualization. 
	1. Binary ![[Pasted image 20240625222228.png|400]]
	2. Sinusoidal![[Pasted image 20240625222502.png|400]]
5. Another nice [video](https://youtu.be/BkyEZwAf-Rw). 
	1. Think of each pair of sin/cos (*embedding* direction) as *clocks* rotating with different frequency. At position 0, all clocks are synced in phase; as position increase (*position* direction), each clock will rotate individually in multiples of delta position.

### Rotary Positional Encoding (ROPE)
1. Rotary Positional Encoding, [RoFormer (arxiv.org)](https://arxiv.org/pdf/2104.09864)
	1. Positional info
		1. not added to the values, so segregation between positional and contextual info
		2. is injected at every layer, not just the initial one
	2. One rotary matrix $\mathbf{R}_{\Theta, m}^d$ for each position $m$. Therefore, *different rotation angles in different positions and different dimensions.*
		1. very similar to the intuition of matrix $M$ above.
	3. In attention layer, only relative rotary $\mathbf{R}_{\Theta, n-m}^d$ (from query-key ${q}_m^{\intercal}\ k_n$) matters.
	4. Nice [post](https://blog.eleuther.ai/rotary-embeddings/) with clean implementation. 
2. Extend context window size beyond that during pre-training
	1. Position Interpolation [2306.15595 (arxiv.org)](https://arxiv.org/pdf/2306.15595)
		1. interpolate position index to match context window size during pre-training; turns out to be better than extrapolation.
	2. [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (arxiv.org)](https://arxiv.org/pdf/2402.13753)
3. implementation
	1. https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
		1. create `theta` for each dimension; create `idx` for each position
		2. calculate their outer product to get the rotary matrix; save it as `buffer`
		3. apply on to `q` and `k` vectors as an extra layer

### My comment
1. Is $\text{sin}(\frac{1}{10000^{2k/d}}t)$ optimal? The low dimensions are quite dense and high dimensions are very sparse.
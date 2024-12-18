# Flash attention
Tags: #Transformers  #FlashAttention 
Authors: Tri Dao
Paper link: 
Talk link: 
1. https://youtu.be/FThvfkXWqtE
Related: 
### Why interesting
1. Author of original paper
### Ideas/conclusions
1.  FLOP vs Memory IO
	1. The latter consumes more time in attention computation
2. Solution
	1. Tiling
		1. block-wise computation, never materialize the full $N*N$ matrix
	2. Recomputation
		1. do not save intermediate result; recompute during back-propagation

# How FlashAttention Accelerates Generative AI Revolution
Tags: #Transformers  #FlashAttention 
Authors: Jia-bin Huang
Paper link: 
Talk link: 
1. https://youtu.be/gBMO1JZav44
Related: 
### Why interesting
1. Very good explanation with derivation and visualization
### Ideas/conclusions
1. Block-wise matrix multiplication (Tiling)
2. How to calculate output $o_i$ one at a time.
	1. Target query $q \in \mathbb{R}^{d_q}$ in $Q\in\mathbb{R}^{N\times d_q}$.
	2. For $k_i, v_i$ pair in $K\in\mathbb{R}^{N\times d_k}, V\in\mathbb{R}^{N\times d_v}$:
		1. $x_i\in \mathbb{R}^1$ is the weight between $q$ and key $k_i$
		2. $o'_i\in\mathbb{R}^{d_v}$ is intermediate weighted sum of the all $v_j$'s where $j\le i$. 
		3. Only when $x_N$, i.e., $i=N$, is calculated do we have one completed weighted value according to $q$.
		4. ![[Pasted image 20241216215430.png|600]]
	3. Here each tile can be multiple rows/cols. ![[flash-attention-jia-bin.mp4]]
# References on GPU and its memory
1. https://horace.io/brrr_intro.html
2. https://www.youtube.com/watch?v=-2ebSQROew4

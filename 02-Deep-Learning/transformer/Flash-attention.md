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
	1. ![[Pasted image 20241216215430.png|600]]
	2. ![[flash-attention-jia-bin.mp4]]
# References on GPU and its memory
1. https://horace.io/brrr_intro.html
2. https://www.youtube.com/watch?v=-2ebSQROew4

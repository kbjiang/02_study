Date: 2024-03-11
Date of publish: 
Authors/Speaker: Kaiming He 
Tags: #DL-Overview

Talk link: https://youtu.be/D_jt-xO_RmI
Paper link:
Related: [[Visualization]] (bootcamp part)

### [Kaiming He](https://youtu.be/D_jt-xO_RmI) ^b044e0
1. Big Idea of DL
	1. compose simple modules to learn complex functions
2. CV, RNN and Transformer
	1. all have *weight sharing* and *local connection*. Conceptually similar.
		1. In Transformer, the FFN layer is local to each token and is identically applied.

1. Also the references within are classic papers

![[Pasted image 20240311184738.png|800]]

![[Pasted image 20240311182934.png|800]]

![[Pasted image 20240311183306.png|800]]
- The $Q/K/V$ are parameter-free; they are just calculation, no learning
- Parameters are all feed-forward
	- from $W_{Q/K/V}$ projections and MLP block
	- [[02_study/02-Deep-Learning/transformer/00-note#^d8e221]]
### [Phillip Isola](https://youtu.be/UEJqxSVtfY0)
1. See [[Visualization]]

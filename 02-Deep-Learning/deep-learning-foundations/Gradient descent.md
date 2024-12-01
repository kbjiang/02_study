## UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION
Tags: #DL-fundation  #DL-generalization #SGD 
Authors: Chiyuan Zhang et.al.
Paper link: https://arxiv.org/pdf/1611.03530
Talk link: 
Related: 
### Why interesting
### Ideas/conclusions
1. Optimization does not mean generalization
2. Regularization is helpful, but not essential to generalization
3. Good exploration on SGD leading to minimal norm solution and decent generalization

## Multiple papers by Nathan Srebro
Tags: #DL-fundation  #DL-generalization #SGD 
Authors: Nathan Srebro et.al.
Paper link: 
1. [Implicit Regularization in Matrix Factorization](https://dl.acm.org/doi/pdf/10.5555/3295222.3295363)
2. [Rank, Trace-Norm and Max-Norm](https://home.ttic.edu/~nati/Publications/SrebroShraibmanCOLT05.pdf)
3. [Learning with Matrix Factorizations](https://home.ttic.edu/~nati/Publications/thesis.pdf)
	1. His dissertation, good overview
Talk link: 
1. [Optimization's Hidden Gift to Learning: Implicit Regularization](https://youtu.be/gh9vrvLx7Mo)
	1. Instead of number of parameter, we need better "complexity measure" to measure generalizability, i.e., norms.
	2. About how SGD is implicitly minimizing certain norms
Related: 

## Towards Understanding the Implicit Regularization Effect of SGD
Tags: #DL-fundation  #DL-generalization #SGD 
Authors: Pierfrancesco Beneventano
Paper link: 
1.  [On the Trajectories of SGD Without Replacement](https://arxiv.org/pdf/2312.16143)
Talk link: 
1. [Towards Understanding the Implicit Regularization Effect of SGD](https://www.youtube.com/watch?v=G70dA2tmbu0)
	1. SGD vs GD; 2nd order term leads to diffusion perpendicular to gradient flow
Related: 

## References
1. [[October#^inductive-bias]]
	1. SGD leads to good inductive bias, i.e., minimal nuclear norm or The Occam's Razor.
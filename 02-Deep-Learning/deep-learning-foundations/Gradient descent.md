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
Related: 
### Why interesting
### Ideas/conclusions
1. It's easy to find global minima, i.e., zero training loss, but which ones can generalize.
2. Instead of number of parameter, we need better "complexity measure" to measure generalizability, i.e., norms.
3. About how SGD is implicitly minimizing certain norms

## Towards Understanding the Implicit Regularization Effect of SGD
Tags: #DL-fundation  #DL-generalization #SGD 
Authors: Pierfrancesco Beneventano
Paper link: 
1.  [On the Trajectories of SGD Without Replacement](https://arxiv.org/pdf/2312.16143)
Talk link: 
1. [Towards Understanding the Implicit Regularization Effect of SGD](https://www.youtube.com/watch?v=G70dA2tmbu0)
Related: 
### Why interesting
### Ideas/conclusions
1. Surprisingly, with and without replacement affects SGD quite a lot.
2. SGD vs GD; 2nd order term leads to diffusion perpendicular to gradient flow
3. without replacement, each batch no longer i.i.d., which is consequential


## Why Momentum Really Works ^dg-momentum
Tags: #DL-fundation #SGD 
Authors: Gabriel Goh
Paper link: 
1.  [Why Momentum Really Works](https://distill.pub/2017/momentum/)
Talk link: 
Related: 
1. [[class-note-chapter-6#^GD]]
### Why interesting
1. quite intuitive with enough equations
### Ideas/conclusions
1. To get optimal $\alpha=\frac{2}{\lambda_1 + \lambda_n}$, I need to use $|1-\alpha \lambda_1|=|1-\alpha \lambda_n|$ and get the non-trivial solution
	1. this is Eqn (4) on P346 of Ref 1.
	2. with $\lambda_1 = \lambda_n$ we have no zig-zag, as the gradient points to $x^*$.
2. About momentum
	1. "gradient descent is a man walking down a hill. He follows the steepest path downwards; his progress is slow, but steady. Momentum is a heavy ball rolling down the same hill. The added inertia acts both as a smoother and an accelerator, dampening oscillations and causing us to barrel through narrow valleys, small humps and local minima."
		1. The effect of damping coefficient is two-fold
			1. smoother (damping): the carried-over inertia removes those 90 degree zig-zags. Curvature no longer dictates 
			2. accelerator: optimal learning rate $\alpha \sim \frac{1}{\sqrt{\lambda}}$ comparing to $\alpha \sim \frac{1}{\lambda}$ without momentum. This shows with momentum we can have greater learning rates, therefore faster convergence.
	2. very nice visualization in section *The Critical Damping Coefficient*
		1. The analogy to a dampened spring is very good.
	3. Plot demonstrating momentum from https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
		1. ![[Pasted image 20241210212005.png|400]]
3. The visualization in *Example: Polynomial Regression* is also very good.


## References
1. [[class-note-chapter-6#^GD]]
2. Reconciling modern machine-learning practice and the classical bias–variance trade-off [[October#^inductive-bias]]
	1. SGD leads to good inductive bias, i.e., minimal nuclear norm or The Occam's Razor.
3. UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION [[November#^e1473e]]
	1. Section 5.
4. [A birds-eye view of optimization algorithms](https://fa.bianp.net/teaching/2018/eecs227at/)

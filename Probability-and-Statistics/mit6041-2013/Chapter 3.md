Date: 2024-01-16
Course:
Chapter: 3
Lecture: 
Topic: #Probability #CountinuousRandomVariable 

## Lectures
### Continuous Random Variables
1. Definition and consequences
	1. $X$ is continuous if $\exists f_X$ s.t. $\mathbf{P}(X\in B)=\int_{B} f_X(x)\ dx$.
		1. $B$ is *any subset of the real line* and $f_X$ is the PDF of $X$. 
		2. Note the *central importance* of $f_X$. ![[Pasted image 20240117071941.png|600]]
	2. By this definition, $\mathbf{P}(a \le b) = \mathbf{P}(a < b)...$ 
	3. $f_X(x)$ can be greater than 1. 
2. Notations
	1. $p_X(x), f_X(x)$... 
3. Cumulative Distribution Function $F_X(x)$.
	1. It unifies both discrete and continuous RVs 
	2. mixed distribution... 
	3. Nice example: Geometric CDF vs Exponential CDF ![[Pasted image 20240117210127.png|600]]
4. Conditional probability
	1. $\mathbf{P}(x\le X \le x + \delta | Y \approx y) \approx f_{X|Y}(x|y) \cdot \delta$. Notice the $\approx$ coz $\mathbf{P}(Y=y)$ is actually 0, which is not strictly valid for the purpose of conditioning.
5. If $X$ and $Y$ are independent
	1. $\mathbf{E}[g(X)h(Y)] = \mathbf{E}[g(X)] \mathbf{E}[h(Y)]$. 
	2. $\text{var}[X + Y] = \text{var}[X] + \text{var}[Y]$. 
		1. Notice var is *not linear* so the multiplication of RVs does not apply
6. Good example: the stick at the end of lecture 9

### Lecture 12: Iterated Expectations; Sum of a Random Number of R.V.s
1. The point being *if $\mathbf{E}[X]$ and/or $\text{var}(X)$ is hard to calculate directly, do it in stages by conditioning.*
	1. E.g. 4.17 the stick problem
2. $\mathbf{E}[X|Y]$ is a R.V., *a function of* $Y$
	1. $\mathbf{E}[X|Y = y]$ is a realization; PDF $f_{X|Y}(x|y)$ is a function of $x$.
		1. PDF and expectation are very different!
	2. We can talk about iterated expectations and variance of the former
	3. The bookstore example
3. Law of total variance
	1. It's understanding. The plot of three sections of students from lecture video
	2. *Pay attention to whose PDF is used!* The widgets and crates. Recitation 13.
4. Random number of R.V.s
	1. $H = X_1 + X_2 + ... X_N$ where $N$ is a R.V. *Don't do $\mathbf{E}[H] = \sum \mathbf{E}[X_i]$ but $\mathbf{E}\mathbf{E}[X_1+X_2...X_N|N]$.*


## My comments
1. *The four steps for solving probability problems*
	1. set up sample space $\rightarrow$ probability law $\rightarrow$ identify event of interest $\rightarrow$ calculate
	2. E.g., Buffon's needle.
2. $Z=X+Y$, $X$ and $Y$ independent. We have  $$
		\begin{align}
		\mathbf{P}(Z\le z|X=x)=&\mathbf{P}(X+Y\le z|X=x) \\
		=&\mathbf{P}(x + Y \le z|X=x) \\
		=&\mathbf{P}(x + Y \le z) \quad  \text{due to independence of $Y$ and $X$.} \\
		=&\mathbf{P}(Y \le z - x) \\
		\end{align}
		$$
	1. Notation is important: *can NOT go from line 1 to line 3 directly* because $X+Y$ is not independent of $X$, but $x+Y$ is since $x$ is just a number.
	2. By differentiating both sides we have $f_{Z|X}(z)=f_Y(z-x)$, therefore $f_{X, Z}(x, z) = f_X(x)f_{Z|X}(z|x) = f_X(x)f_Y(z-x)$.
3. Independence $\implies \rho=0$ but not the other way around.

 
## Interesting/Challenging problems and tricks
1. Prob 9, p188. Mixed Distribution.
	1. My wrong solution:
		1. Let $X_i$ as the RV for expected number of submissions for $i$th coupon *after collecting $(i-1)$th coupon*. So the total number should be $X=\sum X_i$. Given $p_{X_i}(x_i)=1/6$ it's easy to get expectation is 36. This is wrong because $p_{X_i}(x_i)$ are not equal--it's less likely to encounter new type of coupons when some types are already collected.
	2. Solution: 
		1. https://youtu.be/6UdLp2gpmcE
		2. Intuitively, $\mathbf{E}[X_1]=1$ which is obvious. Then $X_2$ is just a geometric with $p=5/6$ (one collected, five left) therefore $\mathbf{E}[X_2]=6/5$. Similarly we'll have $\mathbf{E}[X_i]=6/(6-i) \text{ for } i \ge 0$ and the sum is $\mathbf{E}[X]$.
		3. ![[Pasted image 20240118080317.png|800]]
2. Three pieces form a triangle, Lecture 9, recitation problem 4
3. Absent minded prof., Lecture 9, tutorial problem 3
4. Problem set 5, problem 6. See how $\mathbf{P}(B|A)\neq \mathbf{P}(B)$ which means $B$ and $A$ are not independent. This is because the PDF of 2nd toss has already changed.
	1. Conditional probability can be tricky. The condition changes the PDF and the independence. Need to be careful.
5. Recitation 12, problem 2.
6. *Problem set 6*
	1. problem 2, use the graphics to calculate convolution!
		1. Either find the limits (contains $z$) for integration, or use graphics to move across different $z$.
	2. problem 3, see how to model it as a sum of Bernoulli R.V.
	3. Problem set 6, problem 5, part (b)
		1. With $G$ being the total money brought to the meeting, we have $G=M_1 + M_2 + ... M_K$ where $K$ is the number of attending members. 
		2. I *mistakenly* had $G=MK$. This is wrong because this assumes all $M_i$ are the same all time, while in reality they just have the same distribution. 

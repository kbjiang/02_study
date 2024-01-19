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


## My comments
1. *Bernoulli* vs *Binomial* random variables
	1. Former associates with single coin toss, latter $n$ tosses
2. Usually $g(E[x])\neq E[g(x)]$ when $g(x)$ is not linear. 
	1. Careful with reasoning using expectations!
	2. E.g., distance and speed to calculate travel time. $E[D/V]\neq D/E[V]$
3. Example 2.11, Hat problem, https://youtu.be/EObHWIEKGjA?t=2411
	1. $X=\sum_i X_i$ where $X_i$'s are dependent.
	2. Expectation does not care about dependence when the function of RV is linear
	3. Variance involves $X^2 = \sum_i X_i^2 + \sum_{i, j:i\neq j} X_i X_j$.
4. Geometric distribution
	1. *Example 2.17*. Number of toss till first success.
	2. $\mathbf{E}[X|X-1>0]=\mathbf{E}[X-1|X-1>0]+1=\mathbf{E}[X]+1$. The system is memory-less. I.e., if first trail fails, we just wasted one toss and got back to where we started.

 
## Interesting/Challenging problems and tricks
1. Prob 9, p188. Mixed Distribution.
	1. My wrong solution:
		1. Let $X_i$ as the RV for expected number of submissions for $i$th coupon *after collecting $(i-1)$th coupon*. So the total number should be $X=\sum X_i$. Given $p_{X_i}(x_i)=1/6$ it's easy to get expectation is 36. This is wrong because $p_{X_i}(x_i)$ are not equal--it's less likely to encounter new type of coupons when some types are already collected.
	2. Solution: 
		1. https://youtu.be/6UdLp2gpmcE
		2. Intuitively, $\mathbf{E}[X_1]=1$ which is obvious. Then $X_2$ is just a geometric with $p=5/6$ (one collected, five left) therefore $\mathbf{E}[X_2]=6/5$. Similarly we'll have $\mathbf{E}[X_i]=6/(6-i) \text{ for } i \ge 0$ and the sum is $\mathbf{E}[X]$.
		3. ![[Pasted image 20240118080317.png|800]]

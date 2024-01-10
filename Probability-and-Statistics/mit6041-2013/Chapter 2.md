Date: 2024-01-02
Course:
Lecture: 5
Topic: #Probability #RandomVariable 

## Lectures
### Random variable, expectation and variance
1. Random variable $X$
	1. $X$ is a *real-valued function* of the experimental outcome ($\forall \omega \in \Omega$). ![[Pasted image 20240102083733.png|700]]
		1. E.g., $X$ is the height of students, $x=X(\omega)$ is a possible value of $X$.
			1. Think of $\omega$ as the entity and $x$ as the value of the attribute of interest.
			2. The *randomness* comes from outcome $\omega$, not *random* variable $X$!
		2. It's like a mathematical convenience to represent *events*. E.g., In stead of saying "the sum of two rolls" we have a function.
2. Probability mass functions (PMF)
	1. $\color{yellow}p_{X}(x)=\mathbf{P}({X=x})$. 
		1. Notice the cases of *p*. ![[Pasted image 20240102104442.png|700]]
		2. Notice how there's nothing new from previous chapter except new notations.
3. Joint PMF
	1. for one RV and one event.
		1. $p_{X|A}(x)=\mathbf{P}(X=x|A)$ where $A$ is an event. *Notice different notation from multiple RVs.*
	2. for multiple RVs.
		1. $p_{X|Y}(x|y)=\frac{\mathbf{P}(X=x, Y=y)}{\mathbf{P}(Y=y)}=\frac{p_{X,Y}(x,y)}{p_{Y}(y)}$ where $Y$ is another RV. *Notation*!
		2. The RVs associate with the same experiment, i.e., same sample space.
		3. The RVs don't have to be independent
			1. Example 2.11, $X_i$ is equal but not independent from each other.
4. Hat problem
	1. Expectation does not care about independence
	2. Variance does

## My comments
1. *Bernoulli* vs *Binomial* random variables
	1. Former associates with single coin toss, latter $n$ tosses
2. Usually $g(E[x])\neq E[g(x)]$ when $g(x)$ is not linear. 
	1. Careful with reasoning using expectations!
	2. E.g., distance and speed to calculate travel time. $E[D/V]\neq D/E[V]$
3. Proof
	1. $p_{X|B}(x)=\sum\limits_{i=1}^{n}\mathbf{P}(A_i|B)p_{X|A_i\cap B}(x)$ where ${A_i}$ is a partition.
4. Geometric distribution
	1. *Example 2.17*. Number of toss till first success.
	2. $\mathbf{E}(X|X>1)=\mathbf{E}(X+1)$. The system is memory-less. I.e., if first trail fails, we just wasted one toss and got back to where we started.
5. Problems I got wrong
	1. Prob 40, p133. Six kind of coupons with equal probability, what's the expected number of collection before one gets all coupons at least once? 
		1. My wrong solution:
			1. Let $X_i$ as the RV for expected number of submissions for $i$th coupon *after collecting $(i-1)$th coupon*. So the total number should be $X=\sum X_i$. Given $p_{X_i}(x_i)=1/6$ it's easy to get expectation is 36. This is wrong because $p_{X_i}(x_i)$ are not equal--it's less likely to encounter new type of coupons when some types are already collected.
		2. Solution: 
			1. https://youtu.be/3mu47FWEuqA
			2. Intuitively, $\mathbf{E}[X_1]=1$ which is obvious. Then $X_2$ is just a geometric with $p=5/6$ (one collected, five left) therefore $\mathbf{E}[X_2]=6/5$. Similarly we'll have $\mathbf{E}[X_i]=6/(6-i) \text{ for } i \ge 0$ and the sum is $\mathbf{E}[X]$.
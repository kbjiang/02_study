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
3. $\mathbf{E}[X|X>1]=\mathbf{E}[X]+1$ one toss is wasted.

## My comments
1. *Bernoulli* vs *Binomial* random variables
	1. Former associates with single coin toss, latter $n$ tosses
2. Example 1.30 and 1.31
	1. $\sum\limits_{k=0}^{n} \binom{n}{k}=2^n.$ The sum of all numbers of $k$ element subsets $\equiv$ The number of all possible subsets.
3. Usually $g(E[x])\neq E[g(x)]$ when $g(x)$ is not linear. 
	1. Careful with reasoning using expectations!
	2. E.g., distance and speed to calculate travel time. $E[D/V]\neq D/E[V]$
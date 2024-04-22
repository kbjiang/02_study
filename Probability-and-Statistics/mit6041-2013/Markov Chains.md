Date: 2024-04-22
Course:
Chapter: 
Lecture: 
Topic: #Probability #RandomProcess #MarkovChain

## Lectures
### Lecture 16: Markov Chains - I
1. Markov chains
	1. Unlike the Bernoulli and Poisson, the *future depends on the past*. Markov chain in specific: 
		1. Future only depends on present state $X_n$; does NOT matter how it gets to present state.
		2. $p_{ij}$ independent of $X_n$.
	2. *state*: summarizes the effect of the past on the future
2. $n$-Step Transition Probabilities $r_{ij}(n)$
	1. Probability that state after $n$ time periods will be $j$, given current state is $i$.
3. Classification of states
	1. *Accessible*
	2. *Recurrent* vs *Transient*
	

## My comments
1. It's important to distinguish *Distribution* and *Process*. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
2. Compare definitions between *Bernoulli* (p297) and *Poisson* (p310) process. When $\tau \to 0$, the latter is the limiting case of former 
3. To argue if a process is Poisson, just think of the rate of a single arrival in a small time interval $\delta$. Example 6.13, p318.
## Interesting/Challenging problems and tricks
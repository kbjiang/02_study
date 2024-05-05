Date: 2024-04-22
Course:
Chapter: 
Lecture: 
Topic: #Probability #RandomProcess #MarkovChain

## Lectures
### Lecture 16: Markov Chains - I
1. Markov chains
	1. Unlike the Bernoulli and Poisson, the *future depends on the past*. In specific for Markov, it only depends on current state. 
		1. The state of each time step is a R.V., namely $X_i$, which $\in (1, m)$
		2. $p_{i, i+1} = P(X_i, X_{i+1}|X_i) = P(X_i, X_{i+1}|X_i, X_{i-1}, ... X_0)$. How we got to $i$ does not matter.
	2. *state*: summarizes the effect of the past on the future
2. $n$-Step Transition Probabilities $r_{ij}(n)$
	1. Probability that state after $n$ time periods will be $j$, given current state is $i$.
	2. $p_{ij}=r_{ij}(1)$.
### Lecture 17: Markov Chains - II
1. Classification of states
	1. *Accessible*
	2. *Recurrent* vs *Transient*
		1. Recurrent class
2. Periodic states
	1. It's about groups of states. E.g., here all X states goes into a non-X state, therefore these two groups form a periodic chain.![[Pasted image 20240426065757.png]]
	
### Lecture 18: Markov Chains - III
1. Erlang example starting 15:00 of the lecture video
## My comments
1. It's important to distinguish *Distribution* and *Process*. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
2. Compare definitions between *Bernoulli* (p297) and *Poisson* (p310) process. When $\tau \to 0$, the latter is the limiting case of former 
3. To argue if a process is Poisson, just think of the rate of a single arrival in a small time interval $\delta$. Example 6.13, p318.
## Interesting/Challenging problems and tricks
1. Notation $P(\text{transition to }S_2| \text{ in }S_3\text{ \& State Change})$. The *state change* opposed to circling back to itself.
2. Lec 17, Prob set 8, Prob 2. Random walk.
	1. Notice in part (b), the maximum by itself is not Markovian, but (max, current) pair is.
3. Lec 18, Rec 19, 
	1. Prob 1 (e). The conditional probability calculation. 
		1. $P(X_{n+1}=j|X_n=i, A)$  is really $P((X_{n+1}=j|A)|X_n=i)$, where $X_n = i$ is the world we are in.
	2. Prob 1 (h). Since we only care about next visit, which means transitions after that does not matter, therefore we can think of $S_{6-1}$ as absorbing, then the Expected Time to Absorption applies.
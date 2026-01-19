Date: 2024-04-22
Course:
Chapter: 
Lecture: 
Topic: #Probability #RandomProcess #MarkovChain

## Lectures
### Lecture 16: Markov Chains - I
1. Markov chains
	1. Unlike the Bernoulli and Poisson, here the *future depends on the past*. Therefore the concept of a *state*, which *summarizes the effect of the past on the future*.
		1. In specific for Markov, it only depends on current state. 
	2. Need to define the right *state space*. Nice example P355, Example 7.6.
	3. The state of each time step is a R.V., namely $X_i$, which $\in (1, m)$
	4. $p_{i, i+1} = P(X_i, X_{i+1}|X_i) = P(X_i, X_{i+1}|X_i, X_{i-1}, ... X_0)$. How we got to $i$ does not matter.
2. $n$-Step Transition Probabilities $r_{ij}(n)$
	1. Probability that state after $n$ time periods will be $j$, given current state is $i$.
	2. $p_{ij}=r_{ij}(1)$.
### Lecture 17: Markov Chains - II
1. Classification of states
	1. *Accessible*
	2. *Recurrent* vs *Transient*
		1. Recurrent class
	3. dependence on initial state
2. Periodic states
	1. It's about groups of states. E.g., here all X states goes into a non-X state, therefore these two groups form a periodic chain.![[Pasted image 20240426065757.png]]
	
### Lecture 18: Markov Chains - III
1. Erlang example starting 15:00 of the lecture video
## My comments
1. State classification (including periodicity) matters because
	1. only *aperiodic* chains have *steady-state* distribution. See start of Section 7.3.
	2. absorbing state needs to be defined. See Section 7.4.
2. Steady-state distribution, Expected time to absorption and Mean First Passage/Recurrent Time, all have element of Bootstrapping.
## Interesting/Challenging problems and tricks
1. Notation $P(\text{transition to }S_2| \text{ in }S_3\text{ \& State Change})$. The *state change* opposed to circling back to itself.
2. Lec 17, Prob set 8, Prob 2. Random walk.
	1. Notice in part (b), the maximum by itself is not Markovian, but (max, current) pair is.
3. Lec 18, Rec 19, 
	1. Prob 1 (e). The conditional probability calculation. 
		1. $P(X_{n+1}=j|X_n=i, A)$  is really $P((X_{n+1}=j|A)|X_n=i)$, where $X_n = i$ is the world we are in.
	2. Prob 1 (h). Since we only care about next visit, which means transitions after that does not matter, therefore we can think of $S_{6-1}$ as absorbing, then the Expected Time to Absorption applies.
4. Lec 19, Prob set 9, Prob 4.
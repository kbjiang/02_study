Date: 2024-05-08
Course:
Chapter: 
Lecture: 
Topic: #Probability #LawsOfLargeNumber

## Lectures
### Lecture 19: Weak Law of Large Numbers
1. It considers *long sequence* of identical independent R.V.s. 
	1. Sample mean $M_n$ is average of $n$ samples from one sampling process (one experiment)
	2. Expectation $\mathbf{E}[M_n]$ is average of infinite $M_n$
	3. But we usually concerns with $\mathbf{P}(M_n)$ where $M_n$ is a single experiment and ask the likelihood of it being in some interval?
2. Chebyshev Inequality
	1. links expectation and variance with probability
3. The Weak Law of Large Numbers
	1. *Only* deals with sample mean
4. Convergence in Probability however can be about any R.V.
	1. $Y_n$ converges in probability $\nRightarrow$ $\mathbf{E}[Y_n]$ converges too. See Example 5.8, page 272.
5. The example of Y_n = {0, n} at lec. vid. 23:00
6. intuition leads to CLT around 40:30
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
### Notes
- $\alpha_n(a) = 1/n$ for sample average
- non stationary and fixed step size; non converging but ok
- $(R_n-Q_n)$ provides a direction for updating $Q$.
- *sample-average* can only have varying (1/n) step size.

### Exercises
- Exercise 2.1 In $\mathit{\epsilon}$-greedy action selection, for the case of two actions and $\epsilon$= 0.5, what is the probability that the greedy action is selected?
	- Half of the time greedy, half random, 0.5 + 0.5 * 0.5 = 0.75
- Exercise 2.2: Bandit example Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A1 = 1, R1 = 1, A2 = 2, R2 = 1, A3 = 2, R3 = 2, A4 = 2, R4 = 2, A5 = 3, R5 = 0. On some of these time steps the " case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?
	- See table below. On step 4 and 5, random selection definitely occurred, coz the selected action does not have the highest reward. It possibly have occurred at any other steps.
 | t 	| Q(1) 	| Q(2) 	| Q(3) 	| Q(4) 	| A 	| R  	|   	|
|---	|------	|------	|------	|------	|---	|----	|---	|
| 1 	| 0    	| 0    	| 0    	| 0    	| 1 	| -1 	|   	|
| 2 	| -1   	| 0    	| 0    	| 0    	| 2 	| 1  	|   	|
| 3 	| -1   	| 1    	| 0    	| 0    	| 2 	| -2 	|   	|
| 4 	| -1   	| -1/2 	| 0    	| 0    	| 2 	| 2  	|   	|
| 5 	| -1   	| 1/3  	| 0    	| 0    	| 3 	| 0  	|   	|
- *Exercise 2.3* In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.
	- $q_*(a)$ is randomly selected at the beginning of a run; another layer of randomness is added at each time step--its size matters.
	- the reward grows with more steps, coz more tests out of the 2000 found the best true action over time.
	- in the limit of long run, the estimated Q equals to the true value. Therefore they are $91\%\ (\epsilon =0.1)$ and $99.1\%\ (\epsilon=0.01)$. With higher reward, the latter converges slower.
	- what about the case when $\epsilon=0$?
	- 1 vs 1.54
- Exercise 2.4 If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is a weighted average of previously received rewards with a weighting different from that given by (2.6). What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?
$$
\begin{align*}
Q_{n+1} &= Q_n + \alpha_n[R_n - Q_n] \\ 
&= \alpha_n R_n + (1-\alpha_n)Q_n \\
&= \alpha_n R_n + (1-\alpha_n)(\alpha_{n-1}R_{n-1} + (1-\alpha_{n-1}) Q_{n-1}) \\
&= \alpha_n R_n + (1-\alpha_n)\alpha_{n-1} R_{n-1} + (1-\alpha_n)(1-\alpha_{n-1})Q_{n-1} \\
&= \alpha_n R_n + (1-\alpha_n)\alpha_{n-1} R_{n-1} + (1-\alpha_n)(1-\alpha_{n-1})(\alpha_{n-2} R_{n-2} +(1-\alpha_{n-2}) Q_{n-2}) \\
&= \alpha_n R_n + (1-\alpha_n)\alpha_{n-1} R_{n-1} + (1-\alpha_n)(1-\alpha_{n-1})\alpha_{n-2} R_{n-2} +(1-\alpha_{n})(1-\alpha_{n-1})(1-\alpha_{n-2}) Q_{n-2} \\
&= \sum_{i=1}^{n} \prod_{j=i+1}^{n}(1-\alpha_j)\alpha_{i}R_{i} + Q_1\prod_{i}^{n}(1-\alpha_i)

\end{align*}
$$
- Exercise 2.5
- 2.6: The spikes of optimal action percentage always show up at step $t=num\_ of\_bandits + 1$, prior to which the percentage is about $1/num\_of\_bandits$. This makes sense coz the agent iterate through all arms exactly once in the first $num\_of\_bandits$ steps. At each step, it brings down the value of the corresponding Q estimate. At step $t+1$, the true optimal action has the fair chance to stand out, therefore the spike.
- 2.7: Given $o_0=0$ we have $o_1=\alpha$.
$$
\begin{align*}
o_n &= o_{n-1} + \alpha (1-o_{n-1}) \\
&=\alpha + (1-\alpha)o_{n-1} \\
&=\alpha+\alpha(1-\alpha)+(1-\alpha)^2o_{n-2} \\
&=\alpha\sum_{i=0}^{n-2}(1-\alpha)^i + (1-\alpha)^{n-1}o_1 \\
&=\alpha\sum_{i=0}^{n-1}(1-\alpha)^i \\
&=1-(1-\alpha)^n
\end{align*}
$$
this leads to $\beta_n=\frac{\alpha}{1-(1-\alpha)^n}$ and as n grows, $\beta$ gets closer to $\alpha$, that's the *recency-weighted* part. Also, it's easy to show that, from exercise 2.4, there is a $(1-\beta_1)$ term in the product as coefficient of $Q_1$. Since $\beta_1=1$, the whole coefficient goes to zero, therefore $withough initial bias.$
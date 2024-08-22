Date: 2024-03-13
Course:
Chapter: 
Lecture: 
Topic: #Probability #RandomProcess

## Lectures
### Lecture 13: Bernoulli Process
1. Important to distinguish *Distribution* and *Process*!
	1. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
	2. It's helpful to picture an arrival process; for Bernoulli process, it means people arrive at discrete times.
2. Two PMFs are involved in Bernoulli process. 
	1. *Bernoulli*: Arrival/non-arrival in each time slot.
	2. *Geometric*: Arrival time between each arrival.
	3. Therefore two alternative but equivalent descriptions of the process. 
		1. See bottom of P302 and Example 6.4., where by changing view from one definition to another the calculation is very much simplified.
		2. To simulate a Bernoulli process, you either simulate the flipping of coins, or sample interarrival times from geometric dist.
3. Example 6.2, p299. It's important to start counting from *2nd B other than the 1st*. Video https://youtu.be/gMTiAeE0NCw?t=1626.
	1. This is because only AFTER we saw 1st B did we know it's time to start counting for busy period. To get PMF of B, we reason with PMF of Z. No prior knowledge! 
	2. ![[Pasted image 20240317165916.png|600]]
4. The ErLang PDF, $f_{Y_k}(y)$, is to be understood of each $k$ separately. 

### Lecture 14: Poisson process - I
1. The Poisson process is a *continuous-time analog* of the Bernoulli process applies to situations where there is *no natural way of dividing time into discrete periods*.
	1. E.g., when modeling the number of traffic accident, instead of arbitrary unit time (every second, minute or hour), it's preferrable to consider very small time periods and continuous-time model.
	2. *The events do NOT have to be scarce!* Example 6.9 p313 has "Arrivals of customers is a Poisson process with $\lambda=10$ per minute." The scarcity, i.e. $p \to 0$ is only used when deriving the formula and is almost always true when $\delta \to 0$.
2. Consider an arrival process PMF $P(k, \tau)=\mathbf{P}(\text{exactly } k \text{ arrivals during interval } \tau)$
	1. When it's a process, think in terms of probability $\mathbf{P}$. This usually is used as CDF of $\tau$.
	2. When it's distribution, think PMF.
		1. fix $\tau$ and $\sum_{k} P(k, \tau)=1$.
3. Definition of the Poisson Process
	1. $P$ is same for all interval $\tau$.

### Lecture 15: Poisson process - II
1. The fishing example: blah blah
	1. $\mathbf{P}(\text{fish for more than two and less than five hours})$ can be calculated as $\mathbf{P}(0, 2)(1-\mathbf{P}(2, 5))$ or $\int_{2}^{5} f_{Y_1}(t)dt$. Try calculate.
2. The Random Incidence Paradox
	1. Different probabilistic mechanisms: Average interarrival intervals *by arrivals* vs *by intervals*
	2. Bus stop
	3. another "average family size is 4" vs "on average, a person is from a family size of 6".
3. 
## My comments
1. It's important to distinguish *Distribution* and *Process*. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
2. Compare definitions between *Bernoulli* (p297) and *Poisson* (p310) process. When $\tau \to 0$, the latter is the limiting case of former 
3. To argue if a process is Poisson, just think of the rate of a single arrival in a small time interval $\delta$. Example 6.13, p318.
## Interesting/Challenging problems and tricks
1. P328, Pr 6. An interesting alternative way to see splitting Bernoulli process. 
2. P332, Pr 19. Think in terms of $\text{min}\{X1, X2\}$ and $\text{max}\{X1, X2\}$.
3. P335, Pr 22. *Think of Bernoulli $X_i$ as keeping or discarding an event.*
	1. Event is a success for Binomial or an arrival for Poisson.
4. *Competing exponentials*. By seeing exponentials as 1st arrival of Poisson process simplifies problems.
	1. Example 6.15 and 6.16 on P320.
	2. P335 ,Pr 24. Merging of two Poisson processes with different nature. I.e., $N$ and $T$.
		1. My mistake is thinking in terms of how many $N$ can I fit in $T$, instead I should think of them as two competing processes.
	3. Recitation 15, Pr 3. $X, Y, Z$ are independent exponential R.V.s with parameters $\lambda, \mu, \nu$. What is $\mathbf{P}(X<Y<Z)$?
		1. Think of $X, Y, Z$ as 1st arrivals of three Poisson processes. By merging those processes, the R.V.s are "competing" to arrive first.
		2. $\mathbf{P}(X<Y<Z) = \mathbf{P}(X<\text{min}(Y, Z) \cap  Y<Z)$. This is easy to see that the probability of 1st arrival being $X$ is $\mathbf{P}(X<\text{min}(Y, Z))=\frac{\lambda}{\lambda + \mu + \nu}, \text{ and similarly } \mathbf{P}(Y < Z|X \text{ has arrived})=\frac{\mu}{\mu + \nu}$. 
			1. Same result can be calculated as $\int_{0}^{\infty}\int_{0}^{z}\int_{0}^{y}f_Xf_Yf_Zdxdydz$.
5. Lecture 15. The fishing example: blah blah
	1. $\mathbf{P}(\text{fish for more than two and less than five hours})$ can be calculated as $\mathbf{P}(0, 2)(1-\mathbf{P}(2, 5))$ or $\int_{2}^{5} f_{Y_1}(t)dt$. Try calculate.
6. Lecture 15, problem set 7, prob 2 (f). It's cute how the problem 
	1. Treat the failure part separately (conditional probability) and framed it as Sums of a Random Number of Random Variables.
	2. The sum of failure and success is the Pascal of order $m$. 
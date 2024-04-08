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
4. The ErLang PDF, $f_{Y_k}(y)$, is to be understood of each $k$ separately. In other words, $f_{Y_i}(y)$ is independent of $f_{Y_j}(y)$ given $i\neq j$.

### Lecture 14: Poisson process - I
1. The Poisson process is a *continuous-time analog* of the Bernoulli process applies to situations where there is *no natural way of dividing time into discrete periods*.
	1. E.g., when modeling the number of traffic accident, instead of arbitrary unit time (every second, minute or hour), it's preferrable to consider very small time periods and continuous-time model.
	2. *The events do NOT have to be scarce!* Example 6.9 p313 has "Arrivals of customers is a Poisson process with $\lambda=10$ per minute." The scarcity, i.e. $p \to 0$ is only used when deriving the formula and is almost always true when $\delta \to 0$.
2. Consider an arrival process $P(k, \tau)=\mathbf{P}(\text{exactly } k \text{ arrivals during interval } \tau)$
	1. fix $\tau$ and $\sum_{k} P(k, \tau)=1$.
3. Definition of the Poisson Process
	1. $P$ is same for all interval $\tau$.

### Lecture 15: Poisson process - II
1. The fishing example: blah blah
	1. $\mathbf{P}(\text{fish for more than two and less than five hours})$ can be calculated as $\mathbf{P}(0, 2)(1-\mathbf{P}(2, 5))$ or $\int_{2}^{5} f_{Y_1}(t)dt$. Try calculate.
2. Bus random incidence example
	1. another "average family size is 4" vs "on average, a person is from a family size of 6".
3. 
## My comments
1. It's important to distinguish *Distribution* and *Process*. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
2. Compare definitions between *Bernoulli* (p297) and *Poisson* (p310) process. When $\tau \to 0$, the latter is the limiting case of former 
3. To argue if a process is Poisson, just think of the rate of a single arrival in a small time interval $\delta$. For example, the splitting of Poisson process.
## Interesting/Challenging problems and tricks
1. P328, Pr 6. An interesting alternative way to see splitting Bernoulli process.
2. Lecture 15. The fishing example: blah blah
	1. $\mathbf{P}(\text{fish for more than two and less than five hours})$ can be calculated as $\mathbf{P}(0, 2)(1-\mathbf{P}(2, 5))$ or $\int_{2}^{5} f_{Y_1}(t)dt$. Try calculate.
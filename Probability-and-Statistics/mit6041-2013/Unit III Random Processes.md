Date: 2024-03-13
Course:
Chapter: 
Lecture: 
Topic: #Probability #RandomProcess

## Lectures
### Lecture 13: Bernoulli Process
1. Important to distinguish *Distribution* and *Process*!
	1. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
2. Two PMFs are involved in Bernoulli process. 
	1. *Binomial*: Arrival/non-arrival in each time slot.
	2. *Geometric*: Arrival time between each arrival.
	3. Therefore two alternative but equivalent descriptions of the process. See bottom of P302.
		1. To simulate a Bernoulli process, you either simulate the flipping of coins, or sample interarrival times from geometric dist.
3. Example 6.2, p299. It's important to start counting at *2nd B other than the 1st*. Video https://youtu.be/gMTiAeE0NCw?t=1626.
	1. This is because only AFTER we saw 1st B did we know it's time to start counting for busy period. To get PMF of B, we reason with PMF of Z. No prior knowledge! 
	2. ![[Pasted image 20240317165916.png|600]]

### Lecture 14: Poisson process - I
1. $P(k, \tau)$ , fix $\tau$ and $\sum_{k} P(k, \tau)=1$$.
## My comments
1. It's important to distinguish *Distribution* and *Process*. E.g., Poisson distribution does not involve $\tau$ while Poisson process does.
## Interesting/Challenging problems and tricks
1. P328, Pr 6. An interesting alternative way to see splitting Bernoulli process.
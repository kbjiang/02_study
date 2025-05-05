## The Key Equation Behind Probability
1. Nice [video](https://youtu.be/KHVR587oW8I) with very good intuitions
2. Entropy as average surprise
	1. $h(s) = f(p_s)$ where $h$ is the *entropy or the measure of surprise* of the given state $s$, which should be a function of the probability $p_s$.
		1. More rare more surprise, therefore one possibility is $h$ relates to  $1/p_s$ 
		2. Probabilities multiply while surprises add, therefore the $log$
		3. Put together: $H = \sum{p_s}log(1/p_s)$
3. In reality we don't know $p_s$ but only our belief/model $q_s$. 
	1. ![[Pasted image 20250505070603.png|600]]
4. KL divergence
	1. In $H(P, Q)$ there are two sources of surprises
		1. Model $q_s$ is different from true distribution $p_s$
		2. inherent uncertainty of $p_s$, i.e., entropy $H$
	2. Therefore we need to exclude $H$ to show how different $q$ and $p$ are
		1. $D_{KL} = \sum{p_s}log(1/q_s) - \sum{p_s}log(1/p_s) = \sum{p_s}log(p_s/q_s)$.
## Cross entropy vs *log_loss* and *log-likelihood*:
1. in softmax $p(k|x) = \text{exp}(z_k)/\sum \text{exp}(z_i)$, $z_i$ is called *logits* or *un-normalized log-probabilities*.
2. "Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model. For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model"

## Refs:
1. https://machinelearningmastery.com/what-is-information-entropy/
2. https://machinelearningmastery.com/cross-entropy-for-machine-learning/
3. https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
4. See [here](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative) for a proof of KL's non-negative. 
5. Section 7 of [paper](https://arxiv.org/pdf/1512.00567.pdf)
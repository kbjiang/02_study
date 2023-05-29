### Information entropy
1. The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred.
2. Quantification of the amount of information in things like *events, random variables, and distributions*.
	1. for single event: $h(x) = -log(p(x))$. Here $p(x)$ determins the amount of info in this event.
	2. for the distribution: $H(X) =\sum_x p(x)h(x)$, here $p(x)$ is for expectation.
3. 

### Cross entropy and KL divergence
1. Differences:
	1. "… the cross entropy is the average number of bits **needed** to encode data coming from a source with distribution p when we use model q …". In other words, $H(p, p)=H(p)$.
		$$
		H(p, q) = -\sum_x p(x)*log(q(x))
	$$
	2. "... the KL divergence is the average number of **extra** bits needed to encode the data, due to the fact that we used distribution q to encode the data instead of the true distribution p." That's why it's also called "relative entropy". In other words, $KL(p, p)=0$. 
		$$
		KL(p||q) = H(p, q) - H(p) = \sum_x p*log(\frac{p}{q})
	$$
2. Similarity:
	1. The more different p and q are, the higher the value.
	2. Does not matter which of p and q has more information, CE and KL are always non negative.
	3. Neither is symmetrical. I.e., $H(p,q) != H(q, p)$. Easy to see from the definition equation.
3. 
### Cross entropy vs *log_loss* and *log-likelihood*:
1. in softmax $p(k|x) = \text{exp}(z_k)/\sum \text{exp}(z_i)$, $z_i$ is called *logits* or *un-normalized log-probabilities*.
2. "Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model. For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model"
### Refs:
1. https://machinelearningmastery.com/what-is-information-entropy/
2. https://machinelearningmastery.com/cross-entropy-for-machine-learning/
3. See [here](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative) for a proof of KL's non-negative. 
4. Section 7 of [paper](https://arxiv.org/pdf/1512.00567.pdf)
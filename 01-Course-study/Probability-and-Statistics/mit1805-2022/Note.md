[Course | 18.05 | MIT Open Learning Library](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+18.05r_10+2022_Summer/course/)
## Class 10: Introduction to Statistics, Examples, Likelihood, MLE
1. Example 5, MLE of uniform distributions
## Class 11: Bayesian updating with discrete priors
1. $P(D_2 | H_i, D_1) P(H_i|D_1) = P(H_i, D_2 | D_1)$. Think of $P(|D_1)$ as the new universe.
	1. Sum over $i$ to get $P(D_2|D_1)$; normalize by this sum to get posterior $P(H_i|D_1, D_2)$; now we are in $P(|D_2, D_1)$ universe.
2.  ![[Pasted image 20251231190012.png|600]]
### Studio 5
1. liked the studio, but could be biased given it's my first exposure.

## Class 12: Probabilistic prediction and Odds
1. Predictive probabilities
	1. Prior and posterior probabilities are for *hypotheses*
		1. e.g. $P(12\text{-sided}|R_1=3)$, posterior probability of the dice being 12-sided given 1st roll is 3.
	2. prior and posterior predictive probabilities are for *outcomes*
		1. e.g. $P(R_2 = 8|R_1 =3)$, posterior predictive probability of two consecutive dice rolls.
2. Bayes' factor
	1. The ratio between likelihoods of $\mathcal{H}$ and $\mathcal{H}^c$. Its value gives the strength of 'evidence' provided by data. Larger value means in favor of $\mathcal{H}$ and vice versa
	2. It connects prior and posterior odds and is usually easier to calculate than with probs, since denominators are cancelled.
## Class 13: Bayesian updating with continuous priors
1. Prior/posterior probs (probability of hypotheses) and likelihood (prob. of data given hypothesis) can both be either *continuous* (PDF) or *discrete* (PMF).
	1. Example 6 is continuous in parameter/hypothesis $\theta$ but discrete in data $x$. In the table, $\theta$ in *prior* $2\theta d\theta$ is a variable, but a constant in *likelihood* $\theta^2 (1-\theta)$.
2. Do remember to carry $d\theta$ for parameter/hypothesis around!
	1. $dx$ for data is optional given data is fixed
## Class 15: Conjugate priors: Beta and normal
1. Beta function
	1. seems to be just an easy way to read off the constant for PDFs in form of $C\theta^a (1-\theta)^b$
	2. if prior is a Beta distribution, so is the posterior.
2. Conjugate
	1. if prior and posterior distributions are in the same family, then *the family is conjugate priors* for the likelihood.
		1. E.g., Beta prior and binomial likelihood
3. Gaussian priors (section 7.1)
	1. have a beautiful interpretation of weighted average between prior and data.
	2. more data (larger $n$) puts more weight on data
	3. Variance always decreases, i.e., $\sigma_{\text{post}}^2 < \sigma_{\text{prior}}^2$.
### Good problems
1. concept questions of [Solutions from class 15 | Class 15 lecture slides and problems | 18.05 Courseware | MIT Open Learning Library](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+18.05r_10+2022_Summer/courseware/week8/class15-slides-and-problems-wk/?child=last)
	1. note how *So no amount of data will make the posterior non-zero if corresponding prior is zero.*
## Class 17, 18, 19, 20: Null Hypothesis Significance Testing (NHST)
1. Conceptual takeaways
	1. ==Null distribution is about $\bar{x}$, never distribution about $\theta_0$ for frequentists==!
	2. ==Distribution of the test statistic is computed assuming $\mathcal{H}_0$==
		1. ==Section 5.4., example 2.== *Null distribution* is distribution of the $t$ statistic given $\mathcal{H}_0$, which is NOT the distribution of the data; but they have to be related for the test to be meaningful; in this case, population $\mu_0$ is the common factor.
	2. NHST is a framework, with which are multiple tests, e.g., z-test (statistic $z$), t-test ($t$), chi-square test ($\chi^2$)
		1. Tests follow same procedure, just different statistics/null distributions under different assumptions.  
		2. ==Section 5.4., example 1.== for normal data the Studentized mean (statistic; calculated with sample variance) follows a $t$-distribution (null distribution). This is more suitable than $z$ (standardize mean) when we do not know the population variance of the data.
	3. Test statistic
		1. It's a random variable, its value changes with each trial. 
		2. Find the most reasonable/relevant test statistic.
		3. It's distribution under given hypothesis is fixed though. ==rejection area does not depend on trail data, only the definition of test statistic==! Pset 8, P1(e).
	4. $\mathcal{H}_0$ means "nothing noteworthy is going on";  $\mathcal{H}_A$ means "something noteworthy is going on"
	5. Mental picture: example 3, a discrete case.
	6. We do not *accept* $\mathcal{H}_0$, only *fail to reject*.
		1. E.g., when $\mathcal{H}_A$ is "the coin is biased in favor of heads" instead of "the coin is unfair", it became one-sided. Even when we get small number of heads, we *cannot* reject $\mathcal{H}_0$ *in favor* of $\mathcal{H}_A$. 
		2. E.g., Class 17 in-class problems, problem 2. "The fact that we don’t reject $C_1$ in favor of $𝐶_2$ or $𝐶_2$ in favor of $𝐶_1$ reflects ==the *asymmetry* in NHST==. ==The null hypothesis is the cautious choice==. That is, we only reject 𝐻0 if the data is extremely unlikely when we assume 𝐻0. This is not the case for either $𝐶_1$ or $𝐶_2$."
		3. Ideally both hypotheses should be rejected, but this is the limitation of NHST framework.
2. Significance level $\alpha$, power, $p$-value 
	1. "Some analogies" in Section 4.3.
	2. $\alpha$ should be low (i.e., beyond reasonable doubt) like 0.05, while power should be high. 
		1. To have high power, $\mathcal{H}_A$ should be distinguishable from $\mathcal{H}_0$. More data can help, as it leads to lower variance.
	3. $p$-value: Probability of data *at least as extreme as the observed statistic*, given $\mathcal{H}_0$.
		1. Compared to $\alpha$ to decide if the test statistic falls in rejection area. ==It has nothing to do with probability of Type I error==, which is solely decided by $\alpha$.
		2. For two-sided, need to add both sides, e.g. for normal null dist, $p=P(\left|Z\right|>\left|z\right|) = 2*P(Z>\left|z\right|z)$.
	4. Critical value: right tail $P(X>c_a) = \alpha$; Quantile: left tail $P(X<q_a) = \alpha$
		1. $z_\alpha$ for $\text{N}(0, 1)$, $t_\alpha$ for $t$ dist., $c_\alpha, x_\alpha$ for all purpose.
3. Steps for designing a hypothesis test (Section 5)
	1. statistic: any value that can be calculated from the sample.
		1. E.g., z-statistic, chi-square statistic...
4. Student distribution
	1. The t statistic of a normal sample should follow a t distribution ($T$) with the correct degrees of freedom
	2. $T$ has only one parameter ($df$). Should be compared with unit Norm distribution.
5. TODO: other tests
	1. degrees of freedom
		1. Always count the number of data and its constraints.
	2. $t$-test: when $\sigma$ is unknown, as opposed to the case of $z$-test.
	3. chi-square statistic and chi-square distribution $\chi^2(df)$, where $df$ is degrees of freedom.
	4. When to use other tests? ANOVA for checking if more than two groups have same mean? two-sample $t$ test for two groups? $\chi^2$ for fitting the prob. dist.? 
6. [Likelihood principle - Wikipedia](https://en.wikipedia.org/wiki/Likelihood_principle)
	1. for PSet 9, Problem 2
		1. For Bayesian framework, same data leads to same posterior; for NHST, the $p$-value depends on non-observed data ("at least as extreme as observed" depends on who the experiment is designed) therefore can be different for same observed data.
	2. Class 20, 6 *Stopping Rules*. 
		1. It is a good example showing how NHST is NOT consistent with this principle. 
		2. Also shows Bayesian arrives at the same posterior, given same prior and data, because it only knows observed data.
		3. See also Class 20, 8 *The likelihood principle*.
7. Class 20 is very good discussion. Both lecture and in-class problems.

### Interesting problems
1. Class 19, in class problem, concept question 1. We run a two-sample 𝑡-test for equal means, with 𝛼 = 0.05, and obtain a 𝑝-value of 0.04. What are the odds that the two samples are drawn from distributions with the same mean?
	1. Unknown. $p$-value of 0.04 only means $P(\text{data as or more extreme}|H_0)=0.04$, does NOT say anything about $P(H_0​|\text{data})=0.04$, which is a posterior prob (non Frequentist).
	2. ==So significance level really is about the test setup, something like "this test is so effective, that if $\mathcal{H}_0$ is true, it will only falsely reject is 5 percent of the time."== It attests to how effective the test is, or even the requirement defined prior to experimenting.
		1. This is where $p$-value differs; the latter is evidence for hypothesis given the data.
		2. "too rare to ignore" is the alternative to "false rejection". You ==never know== if the data was really because of the $\mathcal{H}_A$ is true or just being rare given $\mathcal{H}_0$. Or, the data has to be rare enough for me to take the risk of false rejection. 
			1. ==See Class 20 in-class problem, Discussion questions "Type I errors Q1/2/3".== 
		3. NHST is not about "belief" in the prob. of $\mathcal{H}_0$ being true, it's about controlling mistakes. A $p$-value of 0.05 does NOT mean $\mathcal{H}_0$ is 5% likely to be true.
		4. You need to decide if you want to reject the null hypothesis. And if you reject, you do something, such as release the drug
		5. When you reject H0​ at level α, you are saying:_“If I were to repeat this procedure many times in situations where H0​ is true, I would falsely reject it at most α of the time.” That is a statement about **the procedure**, not about **the hypothesis**.
			1. See ==Class 20 in-class problem, Board question 2. "Stop!"== as an example. In experiment 3, one take both first 50 trials and possible 2nd 50 trails as part of the same experiment procedure and calculate the type I errors.
2. Multiple tests
	1. PSet 9, Problem 6, (b). If one of the tests has p-value less than 0.01, it is not proper to reject the null hypothesis. This is because the p-value of the entire experiment is greater than 0.01. That is, since we ran 3 tests each with probability 0.01 of a type I error the total probability of type I error is greater than 0.01--it will be close to 0.03. ==With multiple testing the true p-value of the test is larger than the p-value for each individual test.== 
	2. See also Class 20 in-class problem, Board question 2. "Stop!".
	3. I mistakenly thought that "if one test is significant, then logically the null hypothesis can be rejected." ==The point is that I do NOT know the significant test I saw was TRUE.== It could be type I error!
		1. This is why we use ANOVA which considers between group and within group variances.
		
## Class 22: Confidence intervals
1. It's relation with NHST
	1. pivoting
		1. it's basically the same null distribution of $\bar{x}$, just in reference of different means; when $n$ increase, both distribution narrows down.
	2. $1-\alpha$ confidence interval $\Leftrightarrow$ significant level of $\alpha$
	3. It's easier to think of $x$ in terms of *"is it in the non-rejection region?"*. E.g., if significance level $\alpha$ increase, meaning non-rejection region narrows, then confidence interval narrows down as well.
	4. Mental picture
		1. CI as random variable: the *procedure* produce same size intervals for any $\mu_0$.
		2. $z$ CI for mean. The size of CI is constant as $\sigma$ is constant. ![[Pasted image 20260107202339.png]]
		3. Chi square CI for variance. See how the size of CI depending on value of $s^2$ and $\sigma^2$.![[Pasted image 20260107220700.png]]
2. different CIs
	1. Standardized mean $Z \sim \mathcal{N}(0, 1)$ for $\mu$: unknown $\mu$, known $\sigma$
	2. Studentized mean $T \sim(n-1)$ for $\mu$: unknown $\mu$, unknown $\sigma$
	3. Standardized variance $X^2  \sim \chi^2(n-1)$ dist., for $\sigma^2$ : unknown $\mu$, unknown $\sigma$
3. Class 23 Table 4
	1. discrete confidence interval example
	2. CI type I error $\Leftrightarrow$ CI of $\bar{x}$ does not contain $\theta_{\text{true}}$ $\Leftrightarrow$ $\bar{x}$ not in non-rejection region of $\theta_\text{true}$ (b.c. of pivoting) $\Leftrightarrow$ $\bar{x}$ is less than 1-$\alpha$ likely given $\theta_\text{true}$
4. Class 23, 5 Formal view of confidence intervals. A $(1-\alpha)$ CI for $\theta$ is an interval statistic $I_x$ such that 
$$P(I_x \text{ contains } \theta_0\ |\ \theta = \theta_0) = 1-\alpha$$
	for ==all possible values of $\theta_0$==.
		1. it's a statistic because it's computed on $x$; it changes as $x$ changes with each trial
		2. all possible values of $\theta_0$. Think of $I_x$ as a random variable and the *procedure* works the same for any value of $\theta_0$. It's the flip side of NHST where the true value of $\theta_0$ does not matter.
		3. studio9_problem_1a: the margin of error is independent of the true $\theta$ value.
### Interesting problems
1. PSet 10, Problem 5. 
	1. part (b). Good example where low prior balance out $\mu$ inside 95% CI. This is why CI along cannot tell you how likely a hypothesis is.
	2. part (c). Good conceptual questions.

## References
1. Very good lectures on statistics: https://www.tilestats.com/
2. Python Bayesian model-building interface https://bambinos.github.io/bambi/


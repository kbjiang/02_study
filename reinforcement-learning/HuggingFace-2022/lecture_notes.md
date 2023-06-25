## Unit 1
1. Terminology: agent vs env, state vs reward, action, exploitation vs exploration
![[RL_process.jpg]]
2. RL is a computational approach of learning from actions. I.e., the agent learns from env through trail and error.
3. The agent aims to maximize the expected cumulative reward
4. RL process is called Markov Decision Process (MDP). Markovian has no memory?
5. state (complete) vs observation (partial)
6. exploration vs exploitation (type of actions): 
	1. exploration: trying **random** actions in order to *find more information about the env*
	2. exploitation: using what we **already** know of the env to *maximize the reward*
	3. the choice between a known and a new restaurants is a good intuition.
7. The policy $\pi$: the agent's brain, the goal of training
	1. policy-based: teach the agent what to do given the current state; learn policy func directly
			$a = \pi (s)\quad \mathrm{or} \quad \pi(a|s) = P(A|s)$
	2. value-based: teach the agent to get to a more valuable state; learn policy **indirectly**
		1. to train a **value function**, state vs **expected discounted return** of being at that state
		2. edr is the agent can get if it starts in that state, then act according to policy (going to the state with the highest value)
		3. sounds recursive: *how do I know the value of each state in the 1st place?*
8. d

---


### spinning up
1. Trajectory $\tau$ is unique like an id. Therefore we have $R(\tau) = \sum \gamma^t r_t$ and $P(\tau|\pi) \sim P(s_{t+1}|s_t, a_t)\pi(a_t|s_t)$ 
2. `pi_net` is deterministic with multi-dim output. Should not confuse with softmax.
3. Advantage Function shows how good/bad an action is comparing to average: $A^{\pi}(s, a)=Q^{\pi}(s, a)-V^{\pi}(s)$. This explains why $Q(s, a)$ is defined as take an *arbitrary* action $a$ at $s$ and follows $\pi$ ever after. 

## Unit 2
1. Good intuition with mice and cheese in this unit.
2. MC vs TD
	1. both follow the same update rule: $V(s_t) = V(s_t) + \alpha (X_{s_t} - V(s_t))$ where $X$ is the true state value.
	2. MC: $X = G_t$ ; all values are from real experience; it's unbiased.
	3. TD: $X = R(s_{t+1}) + \gamma V(s_{t+1})$ ; it's bootstrapping and biased by initial values. However, asymptotically it approach the true value with more experiences.
3. On bias and variance
	1. see S&B Fig. 6.1. 
	2. Two posts: [post 1](https://www.endtoend.ai/blog/bias-variance-tradeoff-in-reinforcement-learning/) and [post 2](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565explains) explain why MC is high variance while TD is high bias. 
	3. Also discussion [here](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met) and [here](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)
4. 


### Unit 3
1. Nice walk through of Deep Q Algo. Note how
	1. used two NNs to fix the training target
	2. every $C$ step the target NN was updated by the behavior NN.
2. Nice implementation of DQ Algo, and nice repo overall.
	1. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py

### Unit 4 Policy gradient with Pytorch
1. Stochastic policy, parameterize the policy directly. $\pi_{\theta}(s)=\mathbb{P}(A|s, \theta)$
2. Good illustration. ![[Pasted image 20230411220207.png]]
3. Advantages:
	1. Can learn stochastic policy so that no need to implement an exploration/exploitation trade-off by hand. E.g., no epsilon-greedy needed.
	2. works well with aliased states. Aliased states are those states seem or are the same, but require different actions. This could happen when the agent only have partial access to the env. In such cases, deterministic policy won't be able to explore multiple actions. See this [example](https://youtu.be/y3oqOjHilio?t=1465). Also S&B E.g. 13.1.
4. Derivation from $\nabla_\theta J(\theta) = \nabla \sum_{\tau} P(\tau; \theta) R(\tau)$  to  $\nabla_\theta J(\theta) = \sum_{t=0} \nabla_{\theta} \ \log \pi_{\theta}(a_t|s_t)R(\tau)$.
	1. $P(\tau;\theta)=\prod_{t=0} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t))$ and $P(s_{t+1}|s_t, a_t)$ is the *MDP dynamic* and $\pi$ is agent policy.
	2. The step where we can replace *expectation* $\sum P f(x)$ with *sampling* $1/m \sum_{i} f(x^{(i)})$. 
	3. intuition: log probability of taking action $a_t$ changes according to reward $R(\tau)$. Note reward is independent of $t$ therefore every $t$ along this path $\tau$ will get changed.
5. The *loss function* in Policy optimization differs from ML loss function fundamentally (see [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-the-simplest-policy-gradient) for details):
	1. the data distribution depends on the parameters.
	2. it does not measure performance. There's no target value to "getting closer" to; this loss is just for derivation calculation. It's absolute value or it's trend does not mean much.
6. TODO
	1. really nice example on policy gradient: https://youtu.be/cQfOQcpYRzE
	2. Hongyi's intro to RL https://youtu.be/XWukX-ayIrs?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J
	3. [PyTorch probability distributions](https://pytorch.org/docs/stable/distributions.html#)
	4. [openai-spinningup](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
		1. *Don't let the past distract you*. The argument against $R(\tau)$ does not convince me coz the full $R(\tau)$ term follows derivation. It's just an alternative, not a replacement. The argument about less variance (*But how is this better*) at sample estimate makes sense though.
		2. Baseline part is clean.
	5. [jonathan-hui](https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146)
	6. https://github.com/MrSyee/pg-is-all-you-need/tree/master

### Unit 7 Multi-agent learning
1. Nice [video](https://youtu.be/qgb0gyrpiGk) from MATLAB. Other videos in the same RL series look good too!
2. Self-play
	1. Why: the recent self is an opponent with comparable skill level
	2. How to balance training stability and diversity: how often we change opponents, num. of opponents etc.
3. Evaluation in adversarial games
	1. Cumulative reward is not always meaningful; metric is dependent only on the skill of the opponent.
4. Have NOT been thorough. Should do so when revisit.

### Unit 8
1. Understand the clipped surrogate objective function by [visualization](https://huggingface.co/learn/deep-rl-course/unit8/visualize?fw=pt)
	1. Terminology: 
		1. clipped vs unclipped objective, i.e., $r(\theta)$ vs $\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)$.
		2. $t$ is the index of current minibatch; or current training step
		3. $\hat{A}_t$ is the estimated advantage.
			1. If $A_t$ > 0 for an action, it means this action is favorable, w.r.t previous belief/estimation; vice versa	  
		4. probability ratio $r_t (\theta)=\pi_{\theta}(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$. *Old* is last weights update, not necessarily $t-1$.
	2. The effect of $L^{CLIP}$
		1. Avoid drastic change of objective when it's already in our favor. 
			1. Area 4 is already decreasing the probability of taking action associated with negative $A$
			2. Area 5 is already increasing the probability of taking action associated with positive $A$
		2. Me: *only* admit large change when current example is very different from previous belief. 
			1. Area 3 is decreasing the probability of taking action associated with positive $A$
			2. Area 6 is increasing the probability of taking action associated with negative $A$ 
	3. ![[Pasted image 20230624080039.png]]
2. Understand the clipped surrogate objective function by equation.
		$$
	\begin{align}
		L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\text{min}(r_t (\theta) \hat{A}_t ,\text{clip}(r_t (\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] \\
	\end{align}
	$$
	1. Average is taken over *current minibatch*; each sample in the minibatch contributes it's own part of $L$.
	2. in areas 1 and 2, the ratio $r$ is in reasonable range, therefore no clipping and policy is updated as usual. 
	3. in area 4 and 5, return clipped and $\theta$ *independent* therefore the weights is not updated. You can understand it as, *the ratio change is already in the right direction, we do NOT want to push it too hard.* 
		1. E.g., for area 4, probability of taking this sub-expectation action is already significantly low, i.e., $r<1-\epsilon$, we can leave it alone for now.
	4. in area 3, $r < 1-\epsilon$ and $A>0$, $r$ is positive by definition, the unclipped overtakes the clipped and return $r_t(\theta)\hat{A}$. In this case, since this action is better than expected ($A>0$), the resultant gradient change would point to maximizing $r_t(\theta)\hat{A}_t$ , which in turn increase $r_t(\theta)$. Similar reasoning goes with area 6. *In both cases, the SGA will try to pull the ratio into $1\pm \epsilon$*.
3. My thoughts.
	1. It's all about proper objective that inspires desired actions
	2. proper means make favorable actions ($A>0$) more probable ($r>1$) but not by too much; and vice versa.
	3. we could only influence the *direction* of gradient by maximizing objective, so it takes time to train.
4. References
	1. See section 3.4 of this [thesis](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf) for more intuitions. E.g., for area 3: ![[Pasted image 20230625105153.png]]


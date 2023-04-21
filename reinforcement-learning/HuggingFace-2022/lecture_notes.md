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
	4. On bias and variance, see S&B Fig. 6.1. Also discussion [here](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met) and [here](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)
3. ![[Pasted image 20230403144846.png]]


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
4. Derivation from $\nabla_\theta J(\theta) = \nabla \sum_{\tau} P(\tau; \theta) R(\tau)$  to 
5. intuition
	1. probability of paths changes based on reward $R$.
	2. the path itself is not changed, just its probability
### Unit 1
1. Terminology: agent vs env, state vs reward, action, exploitation vs exploration
![[RL_process.jpg]]
2. RL is a computational approach of learning from actions. I.e., the agent learns from env through trail and error.
3. The agent aims to maximize the expected cumulative reward
4. RL process is called Markov Decision Process (MDP). Markovian has no memory?
5. state (complete) vs observation (partial)
6. exploration vs exploitation (type of actions): 
	1. exploration: trying **random** actions in order to **find more information about the env**
	2. exploitation: using what we **already** know of the env to **maximize the reward**
	3. the choice between a known and a new restaurants is a good intuition.
7. The policy $\pi$: the agent's brain, the goal of training
	1. policy-based: teach the agent what to do given the current state; learn policy func directly
			$a = \pi (s)\quad \mathrm{or} \quad \pi(a|s) = P(A|s)$
	2. value-based: teach the agent to get to a more valuable state; learn policy **indirectly**
		1. to train a **value function**, state vs **expected discounted return** of being at that state
		2. edr is the agent can get if it starts in that state, then act according to policy (going to the state with the highest value)
		3. sounds recursive: *how do I know the value of each state in the 1st place?*
8. d
9. 
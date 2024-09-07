1. assignment-1: the last few plots on how different learning step sizes tracks the true Q value differently.
2. what's the limitation of bandits? "The agent is presented with the same situation and each time and the same action is always optimal." 
	1. The agent cares about *only next reward*. I.e., it's always greedy. The $\epsilon$ was just for introducing the concept of 'exploration'. In other words, it has no effect on the future state/reward.
	2. The state of the envirnoment (the true q values) *never changes*. The distribution around true reward was just there for demonstrating the e-e trade-off. 
3. MDP:
	1. Provides a general framework for sequential decision making
	2. its dynamics are defined by a probability distribution
	3. property: Future state and reward only depends on the current state and action. I.e., remembering earlier states would not improve predictions about the future.
4. Bellman Equation
	1. intuition: In everyday life, we learn a lot without getting explicit, positive, or negative feedback. Imagine for example, you are riding a bike and hit a rock that sends you off balance. Let's say you recover so you don't suffer any injury. You might learn to avoid rocks in the future and perhaps react more quickly if you do hit one. How do we know that hitting a rock is bad even when nothing bad happened this time? The answer is that we recognize that the state of losing our balance is bad even without falling and hurting ourselves. Perhaps we had similar experiences in our past when things didn't work out so nicely. In reinforcement learning a similar idea allows us to relate the value of the current state to the value of future states without waiting to observe all the future rewards. We use Bellman equations to formalize this connection between the value of a state and its possible successors.
5. A policy should *only* depends on the current state. Not things like time or previous states. It's rather a restriction on the state than the agent. For example, if an agent depends on the value from last state, then the state should have a feature called 'previous value'.

### Chapter 4
1. Figure 4.1 is explained at the end of Section 4.2.
2. Section 4.6. Nice intuition about generalized policy iteration (GPI): *policy evaluation* and *policy improvement*.
3. see how both the policy evaluation and improvement utilize *nothing but* the Bellman equation (as an update rule)!
	1. $v_{*}$ is the unique solution to Bellman Eqn. Therefore the iterative way converges/works!
	2. *value iteration* is called so because it's $\pi$ independent.
	3. The *Bellman Optimality Eqn* is special: it is independent of $\pi$, as required by consistency. Eqn (3.14) and (3.19)
4. Nice demonstration of policy iteration starting ~5:00 of video "Policy Iteration".
5. Don't confuse the update iteration $k$ with time stamp $k$ in the definition $G=\sum_k \gamma^k R_{k+t+1}$.
6. It's important to ask if $v_{\pi}(s)$ under question is theoretical or iterative. For e.g., 

### Chapter 5
1. Monte Carlo vs DP:
	1. MC: no model $p(s', r|s, a)$, *agent needs to figure out value functions by pure trial and error.* One episode corresponds to one iteration ($k$), but it's not guaranteed to 'sweep' all states. DP assumes full knowledge of the model. It sweeps all states. See pseudo codes in section 4.3 vs 5.1. 
	2. MC: Eqn (5.1) *defines* the greedy policy soly by current action-value function and is independent of future states, i.e., no $V(s')$. DP: the policy improvement algo in section 4.3.
	3. Both fall under GPI framework
	4. No model, no Bellman eqn?
 1. On-policy control
	 1. Agent learns about the policy used to generate the data.
	 2. This strategy learns a near optimal policy (because of the $\epsilon$ exploration)
	 3. $\epsilon$-greedy is one kind of $\epsilon$-soft methods
	 4. proofs on p102 I don't get???
 2. *Off-policy control*
	 1. The policy to be evaluated (value funcs; estimate) and improved (control) is $target$; the policy controls behavior is $behavior$, which can be quite exploratory. Note that $target$ policy is the one being evaluated and improved! For e.g., a greedy target policy may have different $argmax_a$ at each iteration.
	 2. Since $b$ policy is what being observed, we need importance sampling to convert it's returns to that of the target policy.
	 3. https://www.coursera.org/learn/sample-based-learning-methods/lecture/zjkqu/why-does-off-policy-learning-matter
 3. Ex. 5.7.: is it because that the numerator accumulates but after a few episodes the denominator catches up
 4. Importance sampling:
	 1. good video: https://www.coursera.org/learn/sample-based-learning-methods/lecture/XPxPd/importance-sampling
 5. Return vs reward; $G_t$ vs $V(s_t)$ vs $R(s_t)$

### Chapter 6
1. TD(0) is for prediction. I.e., given a policy, estimate its value function. 
2. Comparison between DP, MC and TD on page 120.
	1. THE diff btw MC and TD: *Eq 6.1 and Eq 6.2*
	2. good intuition/example:
		1. https://www.coursera.org/learn/sample-based-learning-methods/lecture/9Dxlq/the-advantages-of-temporal-difference-learning
		2. see how we use guess at $t+1$ to update the guess at $t$. Bootstrapping!
3. Random walk example (Example 6.2 and Fig 6.2, *nonbatch*): MC is optimal only in a limited way; TD is optimal in predicting returns.
4. Illustration of the difference between *batched* MC and TD(0): Example 6.4, p127.
	1. batched MC finds the estimates that minimizes RMS on training set
	2. batched TD finds estimates that would be exactly correct for maximum-likelihood model of the Markov process.
	3. All these diff bc of Eq 6.1 and 6.2
5. Good intuition for GPI (the mouse and cheese). 
	1. https://www.coursera.org/learn/sample-based-learning-methods/lecture/BcMkZ/sarsa-gpi-with-td
6. More about off-policy
	1. Q-learning as in Eq 6.8 is off-policy. $Q(S_{t+1}, A)$ follows a $\epsilon$ greedy policy, while the target is greedy $argmax_a$. 
	2. In Expected Sarsa, which can be either on or off-policy, when target policy becomes greedy, the $\pi(a'|S_{t+1})$ becomes a delta function, and is equivalent to Q-learning.
	3. Sarsa is on-policy, see how A' is chosen from S' follow the same $\epsilon$-greedy policy in box on p130; this is why Example 6.6 says "Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path". 
	4. In contrary, Q-learning learns the optimal target policy, but acts according to $\epsilon$-greedy behavior policy, that's why "occasionally falling off the cliff because of the eps-greedy action selection." 
	5. See also https://www.coursera.org/learn/sample-based-learning-methods/lecture/1OikH/how-is-q-learning-off-policy

### Chapter 8
1. *model-based* methods (e.g., DP) depend on *planning*; *model-free* (e.g., MC) on *learning*. 
2. *planning* is to update the policy with experiences simulated with the model, instead of 'real' interaction with the enviornment. E.g., *random-sample one-step tabular Q-planning* on p161 is using the model to update action-value function. 
3. *Tabular Dyna-Q* on p164 is about ONE time step, NOT one episode! I.e., for each interaction between agent and environment, the model is looped $n$ times. 
4. *Why planning is efficient*: In between real interactions, planning simulates the state and action of agent and update action values as if the action is real.
5. *planning* leverages agent's experience; once the environment changes, agent will need *acting* to gather new/correct experience.
6. Summary of Part I
	1. section 8.13
	2. the graph in this video. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/HAbzj/course-3-introduction

### Chapter 9, function/approximation as opposed to tabular/precise
1. Tabular/look-up model fails when
	1. Too many states
	2. stochastic states, not seeing the same state twice
2. A representation is an ***agent's internal encoding of the state***, the agent constructs features to 
summarize the current input. Whenever we are talking about features and representation learning,  we are in the land of function approximation. BTW, function approximation can never generate precise value functions because of state generalization.
2. not all function approximation methods are equally well suited for RL
	1. need be able to learn online. I.e., data points seen only once
	2. target function can be nonstationary.
3. state aggregation
	1. it speeds up learning at the cost of reducing state discrimination, i.e., states in same group share same value.
	2. Good intuition, a fish in a pond. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/RUksa/coarse-coding
4. Tile coding:
	1. to create features/representation that can both provide good generalization and discrimination. Once set cannot be updated.
	2. the function connecting feature and weights is usually simple, e.g., linear $v(s)=w \cdot x(s)$
	3. one possiblility: $v(s) \doteq \sum_t v(s, t)$ where $t$ is the tiling.
5. NN generates features:
	1. without exclusively relying on hand-crafted mechanisms (e.g., instead of tile coding, one-hot coding is sufficient.)
	2. has its own weights which can be adjusted during training and learn complicate feature/representation
	3. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/wUQqP/non-linear-approximation-with-neural-networks
	4. see how the receptive maps are different for different features; compare with tile coding
6. MISC:
	1. Notice how Fig. 9.1 the stairs are shifted differently against the true value at both ends. This is due to the state distribution $\mu$.
	2. With SGD, $G_t$ as target and appropriate step size, Monte Carlo converges to local minimum (MSVE); if linear func approx is also assumed, MC converges to global minimum (MSVE). There are stronger theoretical guarantees with linear function approximation than with non-linear function approximation.
	3. DL can learn any component within RL: policy, value function, model.... In this chapter, it was used to estimate/learn the value function. I.e., input is state, target/output is value of the state, loss func for regression is $v_{t+1} - v_{t}$.


### Chapter 10
1. Exploration under function approximation
	1. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/SCKKe/exploration-under-function-approximation
	2. Optimistic initialization, tile coding and $\epsilon$-greedy.
2. Average reward
	1. vs episodic and discounted; has its own value/action functions and Bellman equation
	2. why discounted setting not working with continuous and function approximation: see example in following video.
	3. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/DXGwx/average-reward-a-new-way-of-formulating-control-problems

### Chapter 13
1. policy approximation (as opposed to function approximation)
	1. advantages: blah, blah, 
	2. the approximation has to obey $\sum\pi(s, \theta)==1$ , therefore the soft-max and preference $h(s, \theta)$, which can be any function.
	3. $\pi(a|s, \theta)$ changes smoothly under continuous policy parameterization $\rightarrow$ continuity of the policy dependence on parameters $\rightarrow$ policy-gradient methods
	4. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/NdPo0/the-objective-for-learning-policies 
2. Derivation in Section 13.3, from Eq 13.6 to 13.8
	1. $G_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$: a quantity that can be sampled on each time step whose expectation is equal to the gradient.
	2. therefore this quantity at each time step can be used in the stochastic version of the GD.
3. Misc
	1. Eg 13.1. No tiling, so all states share same feature (significant approximation). Therefore they have to take same actions.
	2. p331, "...for only through bootstrapping do we introduce bias and an asymtotic dependence on the quality of the function approximation". *function approximation* refers to value function $v(s, w)$. 
	3. use stochastic sample as unbiased estimator for expectation value. https://www.coursera.org/learn/prediction-control-function-approximation/lecture/gu7hy/estimating-the-policy-gradient
	4. ![[Pasted image 20230318150423.png]]

### Dimensions
1. contiuning or episodic or average reward
2. on or off policy
3. how is the exploration done
4. model or not
5. how is the model learnt


![[Pasted image 20230319170708.png]]
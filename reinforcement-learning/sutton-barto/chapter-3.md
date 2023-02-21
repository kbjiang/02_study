1. *State* and *Reward* belong to environment, *Action* belongs to agent.
2. for Finite Markov Decision Precesses (MDP), the environment's dynamics ($S$ and $R$) are *completely* determined by 
$$
	p(s', r| s, a) \doteq \text{Pr}\{S_t=s', R_t = r | S_{t-1}=s, A_{t-1}=a  \}.
$$
	Every other entities such as $r(s, a)$ and $p(s', s, a)$ can be derived from it.
3. $S$ and $R$ depends *only* on the immediately preceding state and action. That's why they say MDP is 'memoryless'. 
4. 'This is best viewed a restriction not on the decision process, but on the $state$. The state must contain...' ?
5. "The agent-environment boundary represents the limit of the agent's *absolute control*, not of its knowledge." E.g, we (agent) know how a Rubik's cube (env) works but still not able to solve it. However, the rotating (actions) are under our full control. 
6. Fig 3.3. Good intuition
7. "the online nature of reinforcement learning makes it possible to approximate frequently encountered states, at the expense of less effort for infrequently encountered states."
 

### Exercise
3.1 How about a shooting robot? The env consists of the basket and the ball. The agent is a robot hand that throws the ball. The state is the score board, i.e., has the robot scored? The action is the throwing of the ball, defined by force and angle. The reward could be a function proportional to how close the ball gets to meet the basket.
3.2 One such exception would be me (the agent) try to timing the stock market (env), where reward is the profit/loss. If the environment is totally chaotic, it's very hard to get informative feedback to learn from. Another example would be a system that explicitly depends on previous info further than the immediate one.
3.3 I think it depends on the emphasis of the learning/training. If you are a beginner, it makes sense to draw the line at the accelerator, braker etc. If you are a professional and you care about the level of torque, then the line should be where thr rubber meets the road.
3.4 by replacing $p(s'|s,a)$ with $p(s', r|s, a)$ in the original table and removing the rows with zero p, one gets the new table.
3.5. $$
\sum_{s'\in \mathcal{S}^+}\sum_{r\in \mathcal{R}}p(s', r|s, a) = 1, \forall \  s\in \mathcal{S}^+, a\in \mathcal{A}(s)
$$
3.6. $G_t = -\gamma^{T-t-1}$. Since $R_t = 0\ \forall\ t\ge T$ , this is identical to the countinuing formulation, according to section 3.4.
3.7. The reward is scarse such that the robot is not punished by wandering for long. One way to improve is to assign negative reward to each time step before it finding the exit.
3.8. Notice that $G_5=0$, otherwise trivial.
3.9. Trivial
3.10. Trivial
3.11. W/o policy, one has $r(s_t, a) = \mathbb{E}[R_{t+1}| S_t=s, A_t=a] = \sum_r \sum_{s'} r\ p(s', r|s_t, a)$. With policy, $r(s_t, a) = \sum_r \sum_{s'} \sum_a r\ p(s', r|s_t, a) \pi(a|s)$.
3.12. At the first time step, the agent would have $\pi{(a|s)}$ chance of taking action $a$. Therefore, we have
$$v_{\pi}(s)=\sum_{a}\pi(a|s)\ q_{\pi}(s, a)$$
3.13. Again, since after first action the action-value and state-value of the policy should be identical, we need only pay attention to the first time step.
$$
\begin{align*}
q_{\pi}(s,a)&=r(s, a) + \gamma \sum_{s'} p(s'|s, a) \ v_{\pi}(s')) \\
&=\sum_{s'} \sum_{r} p(s', r|s, a) (r + \gamma \ {\pi}(s'))
\end{align*}
$$


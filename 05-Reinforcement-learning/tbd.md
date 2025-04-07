## Monte carlo Tree search (MCTS)
1. Good intuition and graphics. [Reinforcement Learning: AlphaGo](https://youtu.be/4PyWLgrt7YY)
	1. without RL, AlphaGo learns an expert policy model and can already win against amateurs by pure DL generalization
		1. Expert policy estimates the probability dist. of next move, not probs of winning
		2. This builds the tree of all possible paths, i.e., the tree
	2. RL instill planning/looking ahead, which enables it to beat human experts. In reality, AlphaGo uses both expert model (DL) and value functions (RL)
		1. Value function estimates future winning/gain. It controls how deep to search the tree from current node
		2. But it's an estimate therefore exploration, i.e., search of the tree, is also required (RL)
		3. Detail starts at 5:43.
	

## Resources
1. Reinforcement Learning: A Comprehensive Overview by Kevin Murphy. Author of probabilistic machine learning MIT
	1. https://arxiv.org/pdf/2412.05265v2
2. This lecture series. 深度强化学习 by Shusen Wang
	1. https://youtu.be/jflq6vNcZyA?si=puViyPNlWnWYTWvD
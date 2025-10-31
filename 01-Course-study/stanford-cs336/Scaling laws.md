## Lecture 9: Scaling laws 1
1. Scaling laws 
	1. To extrapolate learnings on smaller models to large models
	2. can be the leap from theoretical upper bounds, which is usually quite loose, to more empirical prediction on performance.
2. Model size vs data size (Chinchilla rule)
	1. with fixed computation budget (IsoFLOP), find the point of lowest loss and lcoate the optimal token/parameter ratio. For Chinchilla this is 20:1.
	2. Training efficiency (Chinchilla regime)
	    - Optimal ratio: ~20:1
	    - Reason: Minimizes loss for a fixed compute budget
	3. Inference performance (modern LLMs)
	    - Optimal ratio: 40–200:1
	    - Reason: Improves generalization, data coverage, and long-tail reasoning

## Lecture 11: Scaling laws 2
2. muP
	1. it's about setting good initialization and layer-wise learning rate so that activations are $\Theta(1)$.
	2. It leads to scale-invariant learning-rate
		1. ![[Pasted image 20251030205011.png]]
	3. detailed derivation skipped

| Goal                                        | Optimal Ratio | Reason                                               |
| ------------------------------------------- | ------------- | ---------------------------------------------------- |
| **Training efficiency** (Chinchilla regime) | ~20:1         | Minimize loss for fixed compute                      |
| **Inference performance** (modern LLMs)     | 40–200:1      | Better generalization, coverage, long-tail reasoning |
	2. This ratio has been pushed up dramatically in favor for inference

### Reference
1. Kaplan
2. Chinchilla Hoffman+ 2022
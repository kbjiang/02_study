1. The [transformation](https://youtu.be/-at7SLoVK_I?t=386) of representation is beautiful. 
2. https://youtu.be/-at7SLoVK_I 
3. https://youtu.be/i94OvYb6noo?t=4247
4. https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
6. [Watching Neural Networks Learn](https://www.youtube.com/@EmergentGarden)
	1. A nice comparison between traditional NN and NN uses Fourier features, i.e., $sin(x), cos(x)$ instead of $x$.
	2. The Fourier features are quite good at approximating *low-dimensional* functions.
7. universal approximation theorem https://youtu.be/Ln8pV1AXAgQ
8. Visualize Matrix multiplication and Attention https://pytorch.org/blog/inside-the-matrix/
9. Great resource [Neural Network playground](https://playground.tensorflow.org/)
### [Deep Learning Bootcamp: Phillip Isola](https://youtu.be/UEJqxSVtfY0)
1. What is "deep learning":
	1. *Neural nets*: A class of machine learning architectures that use stacks of *linear transformations* interleaved with *pointwise nonlinearities*
		1. The goal is to find better *representations* of the data
	2. *Differentiable programming*: A programming paradigm where you parameterize parts of the program and let gradient-based optimization tune the parameters
2. How MLP does nonlinear classification
	1. ![[Pasted image 20240319085954.png|800]]
3. The `linear` and `pointwise nonlinear (e.g., Relu)` transformation visualized
	1. Linear mapping does affine transformation (right top), while `ReLU` clamps data to the positive quadrant ![[Pasted image 20240319085849.png|800]]
	2. A nonlinear classification visualization. See how after training the red and blue dots are separated into their own corner 
		1. Every level is two dimensional $\equiv$ every layer has two neurons ![[visualize-nonlinear-classification.mp4]]
4. Universal approximation with NN
	1. functions as sum of bumps
		1. ![[Pasted image 20240320061907.png|800]]
	2. `ReLU` can approximate any bump
		1. ![[Pasted image 20240320061817.png|800]]
	3. linear combination of bumps approximates any function.
		1. ![[Pasted image 20240320061651.png|800]]
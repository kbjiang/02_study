## VII.1 The construction of Deep Neural Networks
1. The initial weights $x_0$
	1. initial variance $\sigma^2$ 
	2. the number of hidden neurons
2. *Continuous piecewise linear* (CPL) functions
	1. Linear for simplicity, continuous to model an unknown but reasonable rule, piecewise to achieve nonlinearity
	2. Think of origami with flat (assuming ReLU) pieces go to infinity
		1. $v_o$ (input) has $m$ components and $A_1v_0+b_1$ (hidden layer 1) has $N$ neurons. It's $N$ linear functions in dim-$m$ space.
	3. [[December#^spline4dl]]

## VII.3 Backpropagation and the Chain Rule
1. Forward-mode differentiation tracks how one input affects every node. Reverse-mode differentiation tracks how every node affects one output.
2. Ref: 
	1. https://colah.github.io/posts/2015-08-Backprop/
	2. http://neuralnetworksanddeeplearning.com/chap2.html
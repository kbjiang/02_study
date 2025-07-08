## Backward()

1. Autograd explained (A helpful [Post](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95#:~:text=is%20not%20needed.-,Backward()%20function,gradients%20are%20then%20stored%20in%20.))
	1. [PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE): 'explain how the PyTorch autograd system works by going through some examples and visualize the graphs with diagrams'
	2. A nice [Doc](https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html) on memorization.
		1. 'The term *layer error* refers to the derivative of cost with respect to a layer’s *input*. '
		2. Nice derivations.  *H* is activation, *Z* is input, *W* is weights. Notice which the derivative is w.r.t.
     - <img src="https://i.loli.net/2021/03/23/Ty2edFA6WrhH8kC.png" alt="image-20210322185316458" style="zoom:50%;" />

2. We could pass `retrain_graph=True` , to all but last `backward`, if were to call `backward` more than once. 
	1. It retains the intermediate gradients on the computational graph;
	2. See discussion [here](https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795).
 3. Right from the creators. http://blog.ezyang.com/2019/05/pytorch-internals/

## resources
1. Very detailed video shows the math and how back-propagation is similar to forward-propagation
	1. https://youtu.be/yI1PNZRmAI4
2. https://youtu.be/VMj-3S1tku0
3. Paperspace tutorial: [PyTorch Basics: Understanding Autograd and Computation Graphs (paperspace.com)](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)
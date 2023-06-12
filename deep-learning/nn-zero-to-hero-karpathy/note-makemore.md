### Lecture 1
1. Big idea
	To build a bigram character level language model. First by *counting* the bigrams and normalize, then achieve the same next character probability distribution by training a one layer NN with *gradient descent*.
2. Parallel between using counting-based and NN-based.
	1. Interpretation of logits as log of counts; this lead to `softmax` naturally.
	2. adding `N` to counts is equivalent to adding `W**2` term to loss function. See this [timestamp](https://youtu.be/PaCmpygFfXo?t=6618)
3. The physical meaning of  `W=torch.randn(27,27)`, where the...
	1. See at this [timestamp](https://youtu.be/PaCmpygFfXo?t=6023), because of one-hot encoding, only the characters has been fed in so far has a non-zero row in `W.grad`.

### Lecture 2
1. Big idea
	Train a MLP and shared bunch of best practices. See also his post on [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
2. Find a good initial learning rate. [Link](https://youtu.be/TCH_1BHY58I?t=2740)
3. PyTorch internals ref http://blog.ezyang.com/2019/05/pytorch-internals/

### Lecture 3
1. Big idea
	1. The statistics of activations and gradients. See this [timestamp](https://youtu.be/P6sfmUTpUmc?t=4451)
	2. Stabilize NN, i.e., roughly unit Gaussian activations throughout the network, with Reasonable initialization ("Kaiming init")  and/or *Batch Norm* layers. Latter is more popular coz the former is tricky.
	3. But Batch Norm couples samples in the same batch, so should be avoid and use Group/Layer Norms???
	4. Nice *diagnostic tools* starting "PyTorch-ifying the code" to the end.  *Worth revisit.*
2. NN initialization
	1. Keep each layer more or less unit normal distributed to avoid large values saturating activations. E.g., large input to `tanh` would leads to zero gradient and kill the neuron.
	2. Turned out proper initialization in terms of forward pass leads to proper backwards propagation, and vice versa.
	3. Since most activations lead to value contraction, for e.g., $\text{tanh}(inf)=1$, the corresponding initialization has a `gain` to keep the magnitude. See [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) and "Kaiming init" part of the video.
 3. However it's less important now to get NN initialization exactly right
	1. residual corrections; batch/layer/group normalizations; better optimizer like Adam
	2. Just do `torch.randn(fin, fout)/sqrt(fin)` as a rule of thumb
3. *Batch normalization*
	1. It's across samples, thus has a shape of `(1, n_hidden)`. This is also why bias in a linear layer before batch norm layer become *USELESS*, coz for a single neuron it's the same for each sample and get cancelled out. See this [timestamp](https://youtu.be/P6sfmUTpUmc?t=3696)
	2. It also has *gain* and *bias* parameter, which provides the freedom to learn about the batches' mean and std. Those two are *trainable*, as opposed to the running mean and std.
	4. Input samples in a minibatch are now coupled through normalization, i.e., get different loss than when fed in one at a time. However, this is *not* necessarily undesirable. See this [timestamp](https://youtu.be/P6sfmUTpUmc?t=3014).
 4. Good sections starting "PyTorch-ifying the code" to the end.  Notebook [link](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)
	 1. nice implementation to go over again.
	 2. nice visualization as diagnostic tools
		 1. shows how to stabilize the statistics of activations/gradients across each layer via adjusting gain
		 2. The change/data ratio should be around 10e-3; can be an indicator for adjusting learning rate.


### Visualization

### On optimizers
1. https://youtu.be/hd_KFJ5ktUc
2. https://www.ruder.io/optimizing-gradient-descent/
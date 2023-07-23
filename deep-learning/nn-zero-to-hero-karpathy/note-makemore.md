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
	1. Train a MLP and shared bunch of best practices. See also his post on [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
	2. Each *column* of weight corresponds to each neuron!
2. Find a good initial learning rate. [Link](https://youtu.be/TCH_1BHY58I?t=2740)
3. PyTorch internals ref http://blog.ezyang.com/2019/05/pytorch-internals/ (tough read!)

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
	3. Input samples in a minibatch are now coupled through normalization, i.e., get different loss than when fed in one at a time. However, this is *not* necessarily undesirable. See this [timestamp](https://youtu.be/P6sfmUTpUmc?t=3014).
 4. Good sections starting "PyTorch-ifying the code" to the end.  Notebook [link](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)
	 1. nice implementation to go over again.
	 2. nice visualization as diagnostic tools
		 1. shows how to stabilize the statistics of activations/gradients across each layer via adjusting gain
		 2. The change/data ratio should be around 10e-3; can be an indicator for adjusting learning rate.		![[Pasted image 20230612182303.png]]![[Pasted image 20230612182315.png]]![[Pasted image 20230612182346.png]]
5. Maverick Meerkat's answer to  https://stats.stackexchange.com/questions/27112/danger-of-setting-all-initial-weights-to-zero-in-backpropagation

### Lecture 4
1. Big Idea: it's all about calculating backpropagation.
2. See lecture IPYNB.
	1. `grad` is always the same shape as `data`
	2. Interesting duality
		1. Broadcast in forward pass leads to a `sum` in affected dimension (multiple parents, single child) in backward pass.
		2. Sum in forward pass leads to a broadcast in affected dimension (routing of parent gradient) in backward pass.
	3. `logit_maxes.grad` is zero. Its value is not supposed to affect loss, therefore its gradient should be zero.
3. A good post on https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b. 
4. A good video on activation functions. https://youtu.be/gYpoJMlgyXA?t=848

### Lecture 6
1. Big idea
	1. Using next char generation, step by step build: self-attention, multiheads, feedforward...
	2. Adding optimization parts: residual connection, LayerNorm, dropout
	3. Can skip `bigram` part when revisit
2. Good intuitions
	1. `Key` is the token's brand, `Query` is what it wants and `Value` is what it offers. It's the "communication" between tokens.
	2. `FeedForward` is on token level and done independently and identically on each position. It's the token individually "thinking" about what it now has. [timestamp](https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5162) 
		1. In original paper, "While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1."
	3. Each attention block has two parts: 
		1. `attention` is for *communication* (cross tokens)
		2. `ffwd` is for *computation* (single token).
	4. Think of *residual pathway* $x$  as a gradient highway at initialization (before *residual block* $\Delta x$ becomes meaningful.) [timestamp](https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5320)
3. LayerNorm
	1. Normalizes along the embedding dimension of input of a hidden layer. This makes sense, coz *input is the gradient of weights during back propagation*.
	2. [timestamp](https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5573)

### Lecture 6.a ([implementation](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py))
1. In `GPTLanguageModel.forward()` , we flatten `logits` and `targets` when calculating the loss.
2. In `GPTLanguageModel.generate()` 
	1. We only care about the last position of `logits`. That's the position with most/all context information.
	2. Good place to imagine how a sequence of tokens (batch size 1) go through `Head.forward()`.
3. In `Head`
	1. The `:T` in `self.tril` is necessary since at the beginning of generation the length of sequence is smaller than `block_size`.
	2. `W_q` is trainable parameter, `Q=W_q*x` and `wei` are the intermediate variable that are data dependent.
4. Two take aways
	1. *Understand the Transformer architecture in a modular way*
			1. The attention part focus on communication between tokens and *preserves input shape*.
			2. The feedforward part focus on transformation of single token representation.
			3. If think of `Block` (model level) or `Head` (block level) as a black box, its output is just a representation of input data with same shape.
	3. On sequence or on token
		1. `Head.wei` and `positional_embedding` are the only parameters of the model cares about sequence (`:T`), others are all on token level.


### On optimizers
1. https://youtu.be/hd_KFJ5ktUc
2. https://www.ruder.io/optimizing-gradient-descent/
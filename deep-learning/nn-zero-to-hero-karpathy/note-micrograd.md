1. Video [link](https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ); Github [link](https://github.com/karpathy/micrograd); Homework [link](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXlZYTI5d1hBQ2VCU3E4a2ZiSnlXS25ZQXYwd3xBQ3Jtc0trbzlfZ0UyeUdHYlFnMl83Q040b2RLUi0zWVE2cE1NNkVDblpfUXJYYU9FRVhUd3FsS1FHZW5PblRqaGNaMDQ3RHBwbnNRdkgyVERiRFBMRE5TZEN6dG5QTDFpV1gtQjMxa0RYcHJfSERuQVpnN1dIQQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN%3Fusp%3Dsharing&v=VMj-3S1tku0) and local solution.
2. Big idea: 
	1. Backpropagation can be done *recursively*. All one need is *local* gradient (parent w.r.t `self`; saved at `self.grad`) and *global* gradient (`.grad` of parent.) 
	2. Since it's recursive, we only care about *immediate* children/parents.
3. Class `Value`
	1. data features `data`, `_prev`  and  `grad`.
		1. `_prev` keeps track of immediate values leading up to `self`. It is used when connecting the gradients backwards.
		```python
		def backward(self):
	        # topological order all of the children in the graph
	        topo = []
	        visited = set()
	        def build_topo(v):
	            if v not in visited:
	                visited.add(v)
	                for child in v._prev:
	                    build_topo(child)
	                topo.append(v)
	        build_topo(self)
		```
		2. `self.grad` stores `d(out)/d(self) * out.grad`, not `d(self)/d(self._prev)`. Similarly, `op` stores the operation on `self`.
	2. functions `op` and `_backward()`.
		1. There is a 1:1 mapping. For each `op`  we should define its `_backward()`. See definition of multiplying two `Value`s below. 
		2.  `+=` since there can be multiple parents. Not to confuse with multiple runs of backpropagation, where `.zero_grad()` should be applied at each run. 
		```python
		def __mul__(self, other):
			other = other if isinstance(other, Value) else Value(other)
			# Need to specify `_prev` when calculate `out`
			out = Value(self.data * other.data, (self, other), '*')

			# Note that `self.grad` stores d(out)/d(self) * out.grad
			def _backward():
				# `+=` since there can be multiple parents.
				self.grad += other.data * out.grad
				other.grad += self.data * out.grad
			out._backward = _backward

			return out			
		```
		 3. Apparently same can be done with Pytorch. [PYTORCH: DEFINING NEW AUTOGRAD FUNCTIONS](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)
		 4. See also [PyTorch Autograd Explained](https://youtu.be/MswxJw-8PvE)
1. Class `nn`
	1. Single neurons are just collections of weights and bias of type `Value`. Once we have neurons, layers and MLPs follow naturally.
	2. Without layers like normalization, each input sample is independent from each other, batching is just a mathematical convenience.
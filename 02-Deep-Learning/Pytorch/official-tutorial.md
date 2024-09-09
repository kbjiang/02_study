## Introduction to PyTorch
1. PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html): `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset`.
	1. collate function

### References
1. [Hooks](https://youtu.be/syLFCVYua6Q)
2. [Autograd](https://youtu.be/MswxJw-8PvE)
3. nanoGPT
4. phi-3v source file in HF


## Python
1. `iter()` is a function that turns any `iterable` collections into an `iterator` object.
	1. it adds a state to the iterable so that iteration is possible
	2. E.g., `iter([1, 2, 3])` and then `next` is what `for` loop did under the hood.
2. `iterable` object must have `obj.__iter__()` which is equivalent to `iter(obj)`
	1. Try run `dir([1, 2, 3])` and find `__iter__`
	2. If trying to write customized iterable object, also need to include `__next__` that complements `__iter__`
		1. if it works with a `for` loop, it's an iterable.
3. `iterable` vs `iterator`
	1. iterator has a state and knows how to find next value
	2. `i_obj = iter(obj)` turns `obj` into an iterator. There is `__next__` for `i_obj`.
4. Generators are iterators as well.
	1.  `__iter__` and `__next__` are automatically generated for generators
	2. Generators are much easier than customized iterable class. See `MyRange` vs `my_range` example in [video](https://youtu.be/jTYiNjvnHZY?t=692)
		```python
		class PowTwo:
		    def __init__(self, max=0):
		        self.n = 0
		        self.max = max
		
		    def __iter__(self):
		        return self
		
		    def __next__(self):
		        if self.n > self.max:
		            raise StopIteration
		
		        result = 2 ** self.n
		        self.n += 1
		        return result
		``` 
		vs
		```python
		def PowTwoGen(max=0):
		    n = 0
		    while n < max:
		        yield 2 ** n
		        n += 1
		```
	1. And it does not have to end!
		```python
		import time
		def my_range(start):
		    current = start
		    while True:
		        time.sleep(1)
		        yield current
		        current += 1
		
		nums = my_range(1)
		for num in nums:
		    print(num)
		```
1. Ref
	1. https://youtu.be/jTYiNjvnHZY
	2. https://www.programiz.com/python-programming/iterator
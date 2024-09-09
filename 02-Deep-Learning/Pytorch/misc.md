## Tensor
1. increase dimensionality
	1. `torch.Size` reports from outmost dim to inmost dim.
   ```python
   x = tensor([1, 2])
   x[:, None, None] # torch.Size([2, 1, 1])
   x[None, :, None] # torch.Size([1, 2, 1])
   ```
2. broadcast: try it! And note the use of `None` in numpy.
   ```python
   mask = np.array([0,1,0,2])
   print(mask == [1])
   print('Same dim, elementwise. \n')
   print(mask == 1)
   print('Broadcast on 1. \n')
   
   print(mask == [[1]])
   print('Broadcast on mask. \n')
   print(mask == [[1], [2]])
   print('Broadcast on mask, which is compared to both [1] and [2]. \n')
   
   print(mask == [[[1]], [[2]]])
   print('Broadcast on mask, which is compared to both [1] and [2], and dimension is increased. \n')
   print(mask == np.array([1, 2])[:, None, None])
   print('Same as previous case, using numpy and None. \n')
   
   print(mask == [1, 2])
   print('This will fail. \n')
   print(mask == [[1, 2]])
   print('This will fail. \n')
   ```
3. `torch.where(a,b,c)`. This is the same as running the list comprehension `[b[i] if a[i] else c[i] for i in range(len(a))]`
4. [How to index/slice the last dimension of a PyTorch tensor/numpy array of unknown dimensions - Stack Overflow](https://stackoverflow.com/questions/60406366/how-to-index-slice-the-last-dimension-of-a-pytorch-tensor-numpy-array-of-unknown)
5. `[..., -1]` vs `[:, -1]`
6. [python - PyTorch - What does contiguous() do? - Stack Overflow](https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do)
   1. Watch out when doing `transpose` or   `permute` . However, `view` looks fine.
   2. Wait till error occurs to add `contiguous`
7. Assign values to different indices on different rows
	1. use `F.one_hot()` to pick positions
	2. then times array of values
8. Placeholder for understading troch.nn.Parameter
   1. [python - Understanding torch.nn.Parameter - Stack Overflow](https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter)
   2. compared with `torch.tensor`, variables created by `Parameter` shows up the list of parameters of the model.

## DataLoader
1. [Collate_fn](https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders): to override the simple `tensor.stack()`, which requires each item is of same size.  
   ```python
   def collate_fn(batch):
       return tuple(zip(*batch))
   
   # sample: (data, label)
   samp0 = (np.random.randint(5, size=(2,2)), {'l':'cat'})
   samp1 = (np.random.randint(5, size=(1,2)), {'l':'dog'})
   data = [samp0, samp1]
   
   # with and without collate_fn
   dl0 = DataLoader(data, batch_size=2, collate_fn=collate_fn)
   dl1 = DataLoader(data, batch_size=2)
   
   # this returns
   # ((array([[2, 1], [3, 2]]), array([[1, 4]])), ({'l': 'cat'}, {'l': 'dog'}))
   for ds in dl0:
       print(ds)
   
   # this will fail
   # RuntimeError: stack expects each tensor to be equal size, but got [2, 2] at entry 0 and [1, 2] at entry 1
   for ds in dl1:
       print(ds)
   ```

## CUDA
1. release CUDA cache, i.e., last batch of training data removed, but keep the model. Read more [here]([Memory Management, Optimisation and Debugging with PyTorch (paperspace.com)](https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/)).
	```python
	torch.cuda.empty_cache()
	```

## Resources
1. Paperspace tutorial: [PyTorch Basics: Understanding Autograd and Computation Graphs (paperspace.com)](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)
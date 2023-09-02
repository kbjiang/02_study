### My understanding
1. Why `fp16`? Coz not all operations need *single precision (32bit)* as shown in the graph. ![[Pasted image 20230831071818.png]]
2. ![[Pasted image 20230831071909.png]]
3. ![[Pasted image 20230829223457.png]]
4. Two major ideas:
	1. casting
	2. scaling
5. https://youtu.be/9tpLJpqxdE8
	1. fp16 model
	2. fp16 gradient, fp32 master copy
	3. fp16 loss => fp32 loss, 16:28, how scaled fp 32 loss to scaled fp16 gradients?
### Refs
1. Nice [post](https://blog.csdn.net/qq_35985044/article/details/108285982)
2. Nice intro [video](https://youtu.be/b5dAmcBKxHg)
	1. good example of `fp16` fail to capture small update when ratio is high
		```python
		param = torch.cuda.HalfTensor([1.0])
		update = torch.cuda.HalfTensor([.0001])
		print(param + updata)
		# tensor([1.], device='cuda:0', dtype=torch.float16)
		```
1. Pytorch
	1. `torch.amp.autocast` and `torch.amp.GradScaler` in action. [script](https://github.com/karpathy/nanoGPT/blob/master/train.py)
	2. [documentation](https://pytorch.org/docs/stable/amp.html#automatic-mixed-precision-package-torch-amp)

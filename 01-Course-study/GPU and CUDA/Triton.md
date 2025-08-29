1. [Time stamp](https://youtu.be/LuhJEEJQgUM?list=TLPQMjcwODIwMjVEfZMeBMdhXQ&t=2203): this is supposed to print the Triton kernel, good starting point
```python
# TORCH_LOGS = "OUTPUT_CODE" python square_compile.py

import torch
def square(a):
   a = torch.square(a)
   return torch.square(a)

opt_square = torch.compile(square)
opt_square(torch.randn(10000, 10000).cuda())
```
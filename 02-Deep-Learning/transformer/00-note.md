### Generic
1. Wang Mutou https://youtu.be/GGLr-TtKguA
	1. very good
2. Lilian Weng https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
3. Hongyi Lee [https://youtu.be/n9TlOhRjYoc](https://youtu.be/n9TlOhRjYoc)
4. This video [https://youtu.be/C4jmYHLLG3A?list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6](https://youtu.be/C4jmYHLLG3A?list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6)
5. Original post [https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
6. Good reference, but too long: [https://e2eml.school/transformers.html](https://e2eml.school/transformers.html)
7. [nanoGPT/transformer_sizing.ipynb at master · karpathy/nanoGPT · GitHub](https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb)
8. Good visualization with dimensions (Sebastian Raschka)
	1. ![[Pasted image 20241110210414.png|600]]

### LayerNorm
1. it's done at the *layer/embedding* dimension; optionally the batch dimension
2. Nice clear example https://youtu.be/G45TuC6zRf4
3. https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

### Visualization
1. 3Blue1Brown [Visualizing Attention, a Transformer's Heart](https://youtu.be/eMlx5fFNoYc)
2. Recommended by Yufan. https://bbycroft.net/llm
3. [Transformer by hand](https://www.linkedin.com/posts/tom-yeh_transformer-aibyhand-deeplearning-activity-7215281110396620800-fGev?utm_source=share&utm_medium=member_desktop)
4. Nice video from [CodeEmporium](https://youtu.be/Nw_PJdmydZY). 

### Position-wise Feed-forward
1. The same FFN is used for every position, i.e., *weight sharing*, just like how a CNN kernel is shared at each location
	1. Token $\Leftrightarrow$ Pixel, embedding $\Leftrightarrow$ channel. 
		1. Then the FFN in Transformers is a kernel of size $1\times1\times d_{\text{model}}\times 4d_{\text{model}}$.
		2. see [[CNN]] for more on convolutional
	2. Andrew Ng [lecture](https://youtu.be/c1RBQzKsDCk) on $1\times 1$ convolution
		1. the $1 \times 1$ filter is actually $1 \times 1 \times 32$, which can be visualized as one neuron with 32 weights; and it is shared at each pixel/token.  
		2. # filters is the # of output channels.![[Pasted image 20240613174011.png|800]]

### How CNN works
1. Best [lecture](https://youtu.be/OP5HcXJg2Aw) on this topic.
	1. Why CNN? 
		1. Coz Fully connected network is too expensive.
		2. Filters are an effective way to reduce number of neurons. Replacing *global* connections to *local* connections.
			1. This can work because of the *inductive bias*. I.e., visual patterns are invariant inside an image.
	2. *How does CNN reduce the expense? I.e., how does it work?*
		1. Receptive Field: realizing that some patterns are much smaller than a whole image
			1. the idea is that a group of neurons (a filter) covers one receptive field and detect if anything interesting within
			2. No convolution yet--it's just multiplication; this naturally connects to Perceptron idea
		2. Parameter Sharing: same feature can show up at different locations
			1. filter/neuron scans input image $\iff$ receptive field of the image being sent through the CNN one at a time 
	3. Do NOT forget the *channel dimension* of filters! E.g., for 2D conv, it's always $(N, C, H, W)$.
		1. Nice image![[Pasted image 20240613183351.png|600]]
2. Best [visualization](https://youtu.be/JB8T_zN7ZC0) from Brandon Rohrer.
	1. Convolutional layer: each feature/filter in it returns a *filtered* version of the input
	2. Fully connected layer: each feature votes for each of the final category
	3. Great four-pixel toy example: [15:14](https://youtu.be/JB8T_zN7ZC0?t=915) - 26:49
		1. ![[Pasted image 20240816093450.png|800]]
		2. A trained neuron *identifies* a certain pattern, when it sees such a pattern it gets *activated*. This is very perceptron.
3. Nice [video](https://youtu.be/Lakz2MoHy6o) showing *implementation* of the Convolutional layer from scratch.
	1. At 7:30, he shows the forward equation of a convolutional layer is a just a generalization of that of FFN, by replacing multiplication with cross-correlation
4. Detailed visualization videos. [CNN e2e](https://youtu.be/JboZfxUjLSk) and [backpropagation](https://youtu.be/z9hJzduHToc). 
5. Nice [visualization](https://youtu.be/eASwKmKYWeo) on feature maps

### What is Convolution?
1. [3Blue1Brown](https://youtu.be/KuXjwB4LzSA)
	1. $\text{convolve}(\text{list 1}, \text{list 2}) \rightarrow \text{another list}$
	2. in image convolution; in polynomial coefficient; relation to FFT
2. [王木头](https://youtu.be/D641Ucd_xuw?list=PLxIHUhMHF8okwhq8poRuiHBChWjkVUHLL)

### How to 
1.  adversarial 
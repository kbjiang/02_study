1. Best [lecture](https://youtu.be/OP5HcXJg2Aw) on this topic.
	1. Why CNN? 
		1. Coz Fully connected network is too expensive.
	2. *How does CNN reduce the expense? I.e., how does it work?*
		1. Receptive Field: realizing that some patterns are much smaller than a whole image
			1. the idea is that a group of neurons (a filter) covers one receptive field and detect if anything interesting within
			2. No convolution yet--it's just multiplication; this naturally connects to Perceptron idea
		2. Parameter Sharing: same feature can show up at different locations
			1. filter/neuron scans input image $\iff$ receptive field of the image being sent through the CNN one at a time 
		3. Nice image![[Pasted image 20230925102133.png]]
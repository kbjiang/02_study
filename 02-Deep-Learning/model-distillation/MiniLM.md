1. "Figure 1 gives an overview of the deep self-attention distillation. The key idea is three-fold. 
	1. First, we propose to train the student by deeply mimicking the self-attention module, which is the vital component in the Transformer, of the teacher’s last layer. 
		1. The student can have a *different hidden layer size $d_{h}$* coz we are comparing attention weights
		2. The student can have any number of layers coz we are only distilling from last teacher layer
	2. Second, we introduce transferring the relation between values (i.e., the scaled dot-product between values) to achieve a deeper mimicry, in addition to performing attention distributions (i.e., the scaled dot product of queries and keys) transfer in the self-attention module. 
	3. Moreover, we show that introducing a teacher assistant (Mirzadeh et al., 2019) also helps the distillation of large pre-trained Transformer models when the size gap between the teacher model and student model is large."
2. Figure 1![[Pasted image 20230725083256.png]]
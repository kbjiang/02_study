- Paper URL https://arxiv.org/pdf/1503.02531.pdf
- https://arxiv.org/pdf/1207.0580.pdf
- [Gist](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/distillation/readme.md): The output probabilities of a trained model give more information than the labels because it assigns non-zero probabilities to incorrect classes as well. These probabilities tell us that a sample has a chance of belonging to certain classes. For instance, when classifying digits, when given an image of digit _7_, a generalized model will give a high probability to 7 and a small but non-zero probability to 2, while assigning almost zero probability to other digits. Distillation uses this information to train a small model better.


## Notes
1. "...knowledge acquired by a large ensemble of models can be transferred to a single small model." 
2. "If the cumbersome model generalizes well because, for example, it is the average of a large ensemble of different models, a small model trained to generalize in the same way will typically do much better on test data than a small model that is trained in the normal way on the same training set as was used to train the ensemble"???
3. "When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate."
4. Why "soft targets", i.e., not raw probabilities? 
	1. "For tasks like MNIST in which the cumbersome model almost always produces the correct answer with very high confidence, much of the information about the learned function resides in the ratios of very small probabilities in the soft targets. For example, one version of a 2 may be given a probability of 10−6 of being a 3 and 10−9 of being a 7 whereas for another version it may be the other way around. This is valuable information that defines a rich similarity structure over the data (i. e. it says which 2’s look like 3’s and which look like 7’s) but it has very little influence on the cross-entropy cost function during the transfer stage because the probabilities are so close to zero"
5. How:
	1. Ref1: logits
	2. this: raise temperature of final softmax, which includes logits as a special case
6. *"Dropout can be viewed as a way of training an exponentially large ensemble of models that share weights."*
7. "Most of the errors are caused by the fact that the learned bias for the 3 class is much too low."???
8. https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/#7-references
9. DistillBert: https://arxiv.org/abs/1910.01108
	1. triple loss (one for cosine similarity) and student initialization with teacher weight.
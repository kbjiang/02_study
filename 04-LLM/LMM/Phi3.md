1. Rotary embedding
## KV caching


## Flash attention
1.  FLOP vs HBM IO
2. Best explanation https://youtu.be/gBMO1JZav44
	1. safe softmax, online softmax
	2. 

### References on GPU and its memory
1. https://horace.io/brrr_intro.html
2. https://www.youtube.com/watch?v=-2ebSQROew4
3. Talk by the author https://youtu.be/FThvfkXWqtE

## How does vision processor fit in


# CLIP
1. how does it load pretrained weights? I.e., `post_init()`
2. the pooling layer and `BaseModelOutputWithPooling`
3. tokenizer vs processor
4. modelwithprojection == get_text_feature ??
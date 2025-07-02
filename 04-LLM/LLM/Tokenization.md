# Let's build the GPT Tokenizer
Authors: Andrej Karpathy
Link: [lecture](https://youtu.be/zduSFxRajkE)
Tags: #Tokenization 

1. Unicode
	1. a mapping of characters (~150k) to integers (code points)
2. byte-level
3. GPT-2 paper
	1. "Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences" 
4. Tokenization is at the heart of much weirdness of LLMs. 
	1. Problems like cannot spell words , worse at non-English languages (e.g. Japanese), bad at simple arithmetic etc.
	2. Interesting Python example [here](https://youtu.be/zduSFxRajkE?t=685). The difference between `gpt2` and `cl100k_base` suggests that grouping of sequences of spaces alone improves models performance in generating Python
### Unicode, UTF-8
1. Unicode encodings
	2. Why cannot be used as is?
```python
text = "端午 hi"
print([ord(x) for x in text])
print(text.encode('utf-8'))
print(text.encode('utf-16'))
print(list(text.encode('utf-8')))
print(list(text.encode('utf-16')))

text = "端"
print([ord(x) for x in text])
print(text.encode('utf-8'))
print(list(text.encode('utf-8')))
# output:
# [31471, 21320, 32, 104, 105]
# b'\xe7\xab\xaf\xe5\x8d\x88 hi'
# b'\xff\xfe\xefzHS \x00h\x00i\x00'
# [231, 171, 175, 229, 141, 136, 32, 104, 105]
# [255, 254, 239, 122, 72, 83, 32, 0, 104, 0, 105, 0]
#
# [31471]
# b'\xe7\xab\xaf'
# [231, 171, 175]
```

### References
1. [A Programmer’s Introduction to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
2. Tokenization visualized https://tiktokenizer.vercel.app/
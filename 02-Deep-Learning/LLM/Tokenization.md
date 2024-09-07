Date: {{date}}
Course: 
Institute: Stanford 
Lecture:
Topic: #Tokenization

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
1. Main ref: Andrej Karpathy's [lecture](https://youtu.be/zduSFxRajkE)
2. [A Programmer’s Introduction to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
3. Tokenization visualized https://tiktokenizer.vercel.app/
4. 
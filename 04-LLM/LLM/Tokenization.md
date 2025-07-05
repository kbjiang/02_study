# Let's build the GPT Tokenizer
Authors: Andrej Karpathy
Link: [lecture](https://youtu.be/zduSFxRajkE)
Tags: #Tokenization #BPE

1. GPT-2 paper
	1. "Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences" 
2. Tokenization is at the heart of much weirdness of LLMs. 
	1. Problems like cannot spell words , worse at non-English languages (e.g. Japanese), bad at simple arithmetic etc.
	2. Interesting Python example [here](https://youtu.be/zduSFxRajkE?t=685). The difference between `gpt2` and `cl100k_base` suggests that grouping of sequences of spaces alone improves models performance in generating Python
3. *Byte* Pair Encoding (BPE)
	1. First encodes Unicode code points into bytes with UTF-8, then find most popular pairs of bytes iteratively, till desired vocab size is reached.
	2. Problem with naive BPE
		1. `dog.`, `dog!`, `dog?` could all be in the vocab (repetitive, waste of vocab) and mixing of semantic and punctuation
		2. solution: `regex...`
	3. Algorithm for building the vocab (high level)
		1. As in `tiktoken`. unicode code points -> bytes -> merge popular pair of bytes
			1. No `unk` or `oov` because it does not exclude rare code points as `sentencepiece` does
		2. As in `sentencepiece`. unicode code points -> bytes -> merge popular pair of bytes
	4. Algorithm in detail (`tiktoken`)
		```python
		def get_stats(ids):
		    counts = {}
		    for pair in zip(ids, ids[1:]):
		        counts[pair] = counts.get(pair, 0) + 1
		    return counts
		
		def merge(ids, pair, idx):
		    new_ids = []
		    i = 0
		    while i < len(ids):
		        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[-1]:
		            new_ids.append(idx)
		            i += 2
		        else:
		            new_ids.append(ids[i])
		            i += 1
		    return new_ids
		
		vocab_size = 276
		num_merges = vocab_size - 256
		
		ids = list(tokens)
		merges = {}
		for i in range(num_merges):
		    stats = get_stats(ids)
		    pair = max(stats, key=stats.get)
		    idx = 256 + i
		    ids = merge(ids, pair, idx)
		    merges[pair] = idx
		```
	5. sentencepiece (a library)
		1. unlike `tiktoken`, which only does inference, `sentencepiece` does training as well
	6. Misc
		1. Modern tokenization rids of preprocessing/normalizations because they are not necessary, nor are concepts like "sentence" or "paragraph". Just treat the text as a large bytes string!
		2. [Quirks of LLM tokenization](https://www.youtube.com/watch?v=zduSFxRajkE&t=6701s). It's worth rewatching
			1. The `.DefaultCellStyle` is a single token in the vocab. It has difficulty with "how many 'l's are there in this word."
			2. Very good explanation on `SolidGoldMagikarp` towards the end of this session. The discrepancy between training datasets used for tokenization and LLM itself leads to tokens whose embedding never got updated during model training, which would throw the model off big time during inference.
	
# Unicode, UTF-8
Tags: #unicode #utf-8
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

# this means "\x80" is not a valid "utf-8"; this could happen during inference
b"\x80".decode("utf-8", errors="replace")
'�'  # replace ment symbol
```

1. [A Programmer’s Introduction to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/) Highly recommended
	1. The Unicode Codespace
		1. *code points* is roughly "characters" from any scripts, such as "丰". 
		2. set of all code points is codespace, there are $16^4\times17\ \text{(num. of planes)}=1114112$ possible code points with $12\%$ actually assigned.
			1. All the $16$ because codespace *index* is hexadecimal, ranging from U+0000 to U+10FFFF.
			2. ![[Pasted image 20250702133032.png|600]]
		3. Plane 0 is also known as the “Basic Multilingual Plane”, or BMP. The BMP contains essentially all the characters needed for modern text in any script, including Latin, Cyrillic, Greek, Han (Chinese), Japanese, Korean, Arabic, Hebrew, Devanagari (Indian), and many more.
	2. Encoding
		1. 

	3. Decoding
		1. Not all byte sequences are decodable; see the [table]()
			```python
			# this means "\x80" is not a valid "utf-8"; this could happen during inference
			b"\x80".decode("utf-8", errors="replace")
			'�'  # replace ment symbol
			```
			```python
			# The issue: bytes([128]) creates b'\x80' which is invalid UTF-8
			print("Single byte 128:", bytes([128]))  # b'\x80'
			
			# This will cause UnicodeDecodeError:
			# b"\x80".decode("utf-8")  # ERROR!
			
			# Correct way to handle Unicode code point 128:
			unicode_char = chr(128)  # Unicode code point 128
			utf8_bytes = unicode_char.encode('utf-8')  # Proper UTF-8 encoding
			print("Unicode 128 as UTF-8:", utf8_bytes)  # b'\xc2\x80'
			
			# This decodes successfully:
			decoded = utf8_bytes.decode('utf-8')
			print("Decoded back:", ord(decoded))  # 128
			
			# Safe decoding with error handling:
			print("Safe decode with errors='replace':", b"\x80".decode("utf-8", errors="replace"))  # �
			```
2. Tokenization visualized https://tiktokenizer.vercel.app/

## References
1. UTF-8 Encoding Rules

| UTF-8 (binary)                        | Code point (binary)     | Range            |
| ------------------------------------- | ----------------------- | ---------------- |
| `0xxxxxxx`                            | `xxxxxxx`               | U+0000–U+007F    |
| `110xxxxx 10yyyyyy`                   | `xxxxxyyyyyy`           | U+0080–U+07FF    |
| `1110xxxx 10yyyyyy 10zzzzzz`          | `xxxxyyyyyyzzzzzz`      | U+0800–U+FFFF    |
| `11110xxx 10yyyyyy 10zzzzzz 10wwwwww` | `xxxyyyyyyzzzzzzwwwwww` | U+10000–U+10FFFF |
- The `x`, `y`, `z`, `w` represent the actual data bits that encode the Unicode code point
- Bytes with pattern `10xxxxxx` are continuation bytes (128-191 or 0x80-0xBF)
- Continuation bytes cannot appear alone - they must follow a lead byte
- Lead bytes have patterns: `110xxxxx`, `1110xxxx`, or `11110xxx`
- ASCII characters (0-127) use single bytes with pattern `0xxxxxxx`
2. https://github.com/youkaichao/fast_bpe_tokenizer/blob/master/tiktoken_explained.ipynb

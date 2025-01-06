# Chapter 2. Text Classification
1. `dataset.set_format(type='pandas')`
2. good examples on using `map` to apply function to the whole dataset. For e.g., you can even run `model(**input)` to get loss with it.
3. Error analysis: `forward_pass_with_label` to examine examples with highest loss

# Chapter 3. Transformer Anatomy
1. `BertViz`
2. *pre-layer normalization* places layer normalization within the span of the skip connections.
	1. ![[Pasted image 20241230200619.png]]

# Chapter 4. Multilingual NER
1. `sentencepiece` and how it deals with whitespace
2. The Anatomy of the Transformers Model Class
	1. Bodies (outputs last hidden states) and heads. Take `BertForSequenceClassification` for example.
		1. With `self.bert = BertModel(config)` as the body and specific `__init__` to add the head
		2. `BertForSequenceClassification` is a child of `BertPreTrainedModel/PreTrainedModel`, not `BertModel`. Here `BertPreTrainedModel` is for initialization and loading weights.
3. `Trainer`
	1. With `model_init`
	2. with `trainer.predict(dataset).metrics["test_f1"]`
4. `agg` with a list of functions and `droplevel` 
```python
    df_tokens.groupby("input_tokens")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)  # Get rid of multi-level columns
    .sort_values(by="sum", ascending=False)
    .reset_index()
```
5. Error analysis: examine tokens with top validation loss
6. `defaultdict`
	1. Very good when the length of the values are not pre-defined
7. loading via `from_pretrained()` and `post_init`. Basically, to get the "*Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at*" warning

# Chapter 10. 
1. Byte-level tokenizer; for e.g., BPE.
	1. 143,859 Unicode characters, 256 Unicode bytes
		1. If only use Unicode characters, the vocabulary would be huge, even before including words, i.e., combinations of Unicode characters.
		2. If only Unicode bytes, the tokenized text would be super long, overhead on memory
	2. A middle-ground solution is to construct a medium-sized vocabulary by extending the 256-word vocabulary with the most common combinations of bytes.
	3. still not fully understand unicode, character and bytes...
2. Good walkthrough on training new tokenizer for Python
	1. Why new tokenizer: domain specific vocab, smaller vocab.
		1. E.g., indentation 'ĊĠĠĠ' is a token in the new tokenizer now
3. Good walkthrough on training new model from scratch
4. Good details on *accelerate*
	1. The step by step
		1. ![[Pasted image 20250105113847.png|800]]
	2. The *Average* 
		1. ![[Pasted image 20250105113512.png]]

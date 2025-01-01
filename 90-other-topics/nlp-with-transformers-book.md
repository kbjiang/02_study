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
4. `agg` with a list of funcs and `droplevel` 
```python
    df_tokens.groupby("input_tokens")[["loss"]]
    .agg(["count", "mean", "sum"])
    .droplevel(level=0, axis=1)  # Get rid of multi-level columns
    .sort_values(by="sum", ascending=False)
    .reset_index()
```
5. Error analysis: examine tokens with top validation loss
6. `defaultdict`
7. loading via `from_pretrained()` and `post_init`. Basically, to get the "*Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at*" warning

# Chapter 6. Summarization
1. Measuring the Quality of Generated Text
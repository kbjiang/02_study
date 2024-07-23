### OpenAI Bootcamp day
1. OpenAI case study. https://youtu.be/ahnGLM-RC1Y?t=910
	1. query expansion: multiple queries in parallel
2. Evaluation: https://youtu.be/ahnGLM-RC1Y?t=1203

### Multimodal RAG from LangChain
1. https://youtu.be/Rcqy92Ik6Uo
2. https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb
3. Good points
	1. Option 3: 
		1. generate image/table summary (img->text) and link with original img/tbl
		2. text only retrieval. Top k: text chunks and img/tbl summaries
		3. include original text, img/tbl in prompt (img -> base64)
		4. ![[Pasted image 20231128081022.png]]
	2. to mitigate the competing between text and img/tbl summary at retrieval
		1. generate summary for text as well, and retrieve exclusively on summaries. Note: use large chunk size like 4k.
		2. separate indices for each modality and then fuse rankings
	3. image size matters to GPT4-V, too large rate limit issue, too small miss out detail info

### GraphRag
1. Paper and post.
2. Good introduction/tutorial [video](https://youtu.be/LF7I6raAIL4)Explanation of how it works starts at timestamp 2:40.
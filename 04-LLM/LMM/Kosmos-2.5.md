Tags: #LMM
Authors: 
Talk link: 
Paper link:  [2309.11419.pdf (arxiv.org)](https://arxiv.org/pdf/2309.11419.pdf)
Related: 

2. takeaways
	1. It *reads/understands* the text and format in images and *translates* them into 'text + bbox' or 'markdown'. 
	2. Claim to have potential for generalizability across a wide range of text image types.
	3. Simpler architecture than before, e.g., LayoutLM.
3. architecture
	1. VIT for image encoding + decoder for text generation ![[Pasted image 20230926145426.png]]
4. Data preparation
	1. text lines: 
		1. $\langle \text{s} \rangle \langle \text{image} \rangle \text{Image Embedding}\langle \text{/image} \rangle \cup_n(\mathbf{B}_n\bigoplus\mathbf{T_n}) \langle /s \rangle$, where $\mathbf{T}$ are text lines, $\mathbf{B}$ are bboxes and $\bigoplus$ means concatenation.
		2. Note that image is the input,  $\mathbf{T}$ and $\mathbf{B}$ targets/predictions. Similar for markdown case.
	2. markdown:
		1. $\langle \text{s} \rangle \langle \text{image} \rangle \text{Image Embedding}\langle \text{/image} \rangle \text{Markdown Text} \langle /s \rangle$
	3. *A lot of processing tips in section 2.4.*
5. loss functions
	1. NED + NTED
		1. NED is string-based similarity, NTED considers hierarchical structure for markdown.
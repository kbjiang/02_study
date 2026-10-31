# Analogy: CNN Channels vs. Multi-Head Attention

Multiple attention heads are conceptually very similar to multiple channels in a CNN. Both mechanisms allow the model to extract different features or look at the data through different "lenses" simultaneously.

## The Direct Analogy

| Feature | CNN Channels | Multi-Head Attention |
| :--- | :--- | :--- |
| **The Analogy** | Different channels detect different spatial features (e.g., one channel finds horizontal edges, another finds textures, another finds color gradients). | Different heads track different relational features (e.g., one head tracks subject-verb agreements, another tracks pronouns, another tracks temporal context). |
| **Parameter Split** | Each output channel has its own distinct set of weight kernels. | Each head has its own set of $W_Q, W_K, W_V$ projection matrices. |
| **Combination** | Output channels are stacked side-by-side to form a new channel dimension. | Head outputs are concatenated side-by-side and then linearly projected back to the original dimension. |
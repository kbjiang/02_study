# LLaVA Paper — Two Key Takeaways

**Primary source:** Liu et al., [Visual Instruction Tuning (LLaVA), arXiv:2304.08485](https://arxiv.org/abs/2304.08485).

## Figure 1 — LLaVA network architecture

![Figure 1: LLaVA network architecture](https://arxiv.org/html/2304.08485v2/x1.png)

*Source: Figure 1, Liu et al., “Visual Instruction Tuning.” The figure is embedded from the paper's arXiv HTML version. [Open the paper](https://arxiv.org/abs/2304.08485).*

### How to read the figure

- **Image pathway:** Image `X_v` → CLIP vision encoder → visual features `Z_v` → projection matrix `W` → visual tokens `H_v`.
- **Text pathway:** Language instruction `X_q` → language tokens/embeddings `H_q`.
- **Fusion and generation:** The LLM processes the visual tokens and language tokens together and generates a language response `X_a`.

The projection layer maps CLIP visual features into the LLM's word-embedding space. It produces a sequence of continuous visual embeddings that the LLM can consume alongside text tokens.

---

## 1. Two-stage training

LLaVA separates visual-language alignment from visual instruction tuning.

| | Stage 1: Pre-training / feature alignment | Stage 2: Visual instruction tuning |
|---|---|---|
| Main objective | Train a compatible visual tokenizer for the frozen LLM | Teach the model to follow visual instructions |
| Vision encoder | Frozen | Frozen |
| Projection layer | Trainable | Trainable |
| LLM | Frozen | Trainable |
| Training data | Image-caption pairs | Image-based instruction-following conversations |

**Key idea:** Connecting vision and language is different from teaching a model to perform multimodal tasks.

- Stage 1 establishes a compatible interface between visual representations and the LLM.
- Stage 2 adapts the model to answer questions and follow instructions about images.

## 2. The projection layer as a compatible visual tokenizer

The LLaVA paper explicitly describes Stage 1 as:

> “This stage can be understood as training a compatible visual tokenizer for the frozen LLM.”

The projection layer plays the role of a visual-to-language adapter:

1. The CLIP vision encoder extracts visual features from an image.
2. The projection layer maps those features into the LLM's embedding space.
3. The resulting continuous vectors serve as visual tokens that the LLM can consume alongside text tokens.

### Text and visual pathways

| Text pathway | Visual pathway |
|---|---|
| Text tokenizer | CLIP vision encoder + projection layer |
| Text token IDs | No discrete visual token IDs are required in this pathway |
| Text token embeddings | Projected visual embeddings |
| LLM | LLM |

**Why “compatible” matters:** The projection layer learns to represent visual information in a format the frozen LLM can process. It does not need to change the pretrained vision encoder or LLM during Stage 1.

---

## Interview-ready summary

1. **Two-stage training:** First establish visual-language representation compatibility; then teach the model to follow visual instructions.
2. **Visual tokenizer:** The projection layer serves as a compatible visual tokenizer, mapping CLIP visual features into the frozen LLM's embedding space.
3. LLaVA is a generative multimodal model that ==conditions an LLM on visual features== to produce text.

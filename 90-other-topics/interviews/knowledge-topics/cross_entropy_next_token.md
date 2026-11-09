# Cross-Entropy Loss for Next-Token Prediction

Cross-entropy loss measures the discrepancy between a language model's predicted probability distribution over its vocabulary and the true target token.

---

## 1. How It Works

1. **Logits Production**
   The transformer's final hidden state passes through the language modeling head to produce unnormalized score vectors called **logits** for every token in the vocabulary.

2. **Softmax Normalization**
   A softmax function converts the logits into a probability distribution $P$ where all token probabilities sum to 1.

3. **True Target (One-Hot)**
   The actual ground-truth next token acts as a one-hot vector $Y$, where the correct token index has a value of 1 and all other indices are 0.

4. **Negative Log-Likelihood Simplification**
   Because the target is a one-hot vector, the general cross-entropy formula collapses to the negative log-probability assigned to the single true token:
   $$\text{Loss} = -\log P(\text{target token})$$

---

## 2. Key Properties

* **Strong Penalty for Confident Mistakes:** If the model assigns a high probability (e.g., 0.90) to the correct token, the loss is near zero. If it assigns a tiny probability (e.g., 0.0002) to the correct token, the negative logarithm shoots up, heavily penalizing the model.
* **Connection to Perplexity:** The exponential of the cross-entropy loss ($\exp(\text{Loss})$) equals the model's **perplexity**, a standard metric used to evaluate how well a language model predicts a sample.
* **Framework Implementation:** Deep learning libraries like PyTorch compute this directly via `nn.CrossEntropyLoss()`, which takes the raw unnormalized logits and the target token index, internally combining log-softmax and negative log-likelihood for numerical stability.

---

## 3. Why Other Tokens Matter (The Softmax Competition)

We **do consider the loss on other tokens**, but its contribution is hidden inside the mathematical mechanics of the softmax normalization step. Because cross-entropy compares the model’s distribution against a target distribution where the target token holds 100% of the probability weight, the formula collapses to a single term.

The model's probability distribution is coupled across the entire vocabulary. The probability for the true target token $y$ is computed as:

$$P(\text{target}) = \frac{e^{z_y}}{\sum_{j \in \text{vocab}} e^{z_j}}$$

Where $z_j$ represents the raw unnormalized logit for token $j$. 

To lower the overall loss, the model must maximize $P(\text{target})$. Because every single incorrect token's logit ($z_j$) lives in the denominator, the model can only increase $P(\text{target})$ by doing two things simultaneously:
* **Pushing up** the logit of the correct token ($z_y$).
* **Pulling down** the logits of all incorrect tokens ($z_j$).

---

## 4. The Gradient Perspective

Looking at the gradient of the cross-entropy loss with respect to the raw logits reveals exactly how the model updates its weights.

### For the correct target token ($y$):
$$\frac{\partial \text{Loss}}{\partial z_y} = P(\text{target}) - 1$$

### For any incorrect token ($i \neq y$):
$$\frac{\partial \text{Loss}}{\partial z_i} = P(\text{incorrect}_i) - 0 = P(\text{incorrect}_i)$$

### How the Weights Learn
* **The Correct Token:** The gradient $(P(\text{target}) - 1)$ is always negative. During gradient descent, subtracting a negative gradient pushes the logit $z_y$ upward.
* **The Incorrect Tokens:** The gradient $P(\text{incorrect}_i)$ is always positive. Subtracting a positive gradient drives these logits downward. The higher the model's confidence was in a wrong token, the larger its positive gradient, and the harder its logit gets slammed down.

While the loss formula looks like it only cares about a single target, the backpropagation step actively modifies the scores of the **entire vocabulary** to suppress the wrong answers.

# LoRA (Low-Rank Adaptation) Summary

LoRA is a parameter-efficient fine-tuning technique that customizes large models without altering their original weights.

## 1. How It Works
* **Frozen Base:** The pre-trained model weights ($W$) are frozen to prevent catastrophic forgetting.
* **Side-Path Matrices:** Two small trainable matrices ($A$ and $B$) are injected into the layers (typically attention).
* **Low-Rank Math:** The weight update $\Delta W$ is approximated by multiplying $B \times A$.
  $$\text{Output} = WX + \Delta WX = WX + (B \times A)X$$
* **Zero Latency:** For production, $B \times A$ can be mathematically merged back into $W$, adding no overhead.

## 2. The Scaling Factor ($\frac{\alpha}{r}$)
* **The Formula:** LoRA scales the adapter output by a constant ratio of `lora_alpha / r`.
* **The Purpose:** When rank ($r$) increases to provide more capacity, the scaling factor automatically decreases. 
* **Benefit:** This stabilizes the magnitude of the updates, allowing developers to change $r$ without having to re-tune the learning rate.

## 3. Asymmetric Initialization
* **Why not both zero?** If both matrices $A$ and $B$ are initialized to zero, their initial gradients evaluate to zero due to symmetry. The weights stay permanently stuck, and the network cannot learn.
* **The Strategy:** 
  * **Matrix A** is initialized with random Gaussian numbers (Kaiming uniform).
  * **Matrix B** is initialized to all zeros.
* **Result:** The overall adapter update starts at exactly zero ($B \times A = 0$), preserving original model behavior on step zero, but breaks symmetry so gradients flow immediately during backpropagation.
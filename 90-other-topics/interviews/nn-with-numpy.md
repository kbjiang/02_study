### 1. Mathematical Definitions

Let $\mathbf{z} = [z_1, z_2, \dots, z_C]^T$ be the pre-activation logit vector for $C$ classes.

#### The Softmax Activation
The activated probability $\hat{y}_k$ for class $k$ is defined as:
$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}$$

#### The Categorical Cross-Entropy Loss
For a true one-hot encoded target vector $\mathbf{y} = [y_1, y_2, \dots, y_C]^T$, the loss $\mathcal{L}$ is:
$$\mathcal{L} = -\sum_{c=1}^C y_c \ln(\hat{y}_c)$$

---

### 2. Step-by-Step Chain Rule Derivation

We want to find how the loss changes with respect to any arbitrary logit $z_i$, which is $\frac{\partial \mathcal{L}}{\partial z_i}$. Because every single logit $z_i$ influences the denominator of *every* output probability $\hat{y}_c$, a change in $z_i$ affects all terms in the loss summation. 

By the multivariate chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{c=1}^C \frac{\partial \mathcal{L}}{\partial \hat{y}_c} \frac{\partial \hat{y}_c}{\partial z_i}$$

#### Step A: Derivative of Loss w.r.t Output Probability ($\frac{\partial \mathcal{L}}{\partial \hat{y}_c}$)
Differentiating $\mathcal{L}$ with respect to a specific predicted probability $\hat{y}_c$:
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_c} = -\frac{y_c}{\hat{y}_c}$$

#### Step B: Derivative of Softmax w.r.t Logits ($\frac{\partial \hat{y}_c}{\partial z_i}$)
We must consider two distinct geometric cases here because of how indices interact in the quotient rule:

*   **Case 1: $c = i$ (Derivative of a class probability w.r.t its own logit)**
    Using the quotient rule $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$:
    $$\frac{\partial \hat{y}_i}{\partial z_i} = \frac{e^{z_i}\left(\sum_{j=1}^C e^{z_j}\right) - e^{z_i}\left(e^{z_i}\right)}{\left(\sum_{j=1}^C e^{z_j}\right)^2}$$
    $$\frac{\partial \hat{y}_i}{\partial z_i} = \left(\frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}\right) \left(\frac{\sum_{j=1}^C e^{z_j} - e^{z_i}}{\sum_{j=1}^C e^{z_j}}\right) = \hat{y}_i (1 - \hat{y}_i)$$

*   **Case 2: $c \neq i$ (Derivative of a class probability w.r.t a different logit)**
    Here, the numerator $e^{z_c}$ treats $z_i$ as a constant (derivative is 0):
    $$\frac{\partial \hat{y}_c}{\partial z_i} = \frac{0 \cdot \left(\sum_{j=1}^C e^{z_j}\right) - e^{z_c}\left(e^{z_i}\right)}{\left(\sum_{j=1}^C e^{z_j}\right)^2}$$
    $$\frac{\partial \hat{y}_c}{\partial z_i} = -\left(\frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}\right) \left(\frac{\sum_{j=1}^C e^{z_i}}{\sum_{j=1}^C e^{z_j}}\right) = -\hat{y}_c \hat{y}_i$$

---

### 3. Combining the Terms

Now we substitute our results back into our main chain rule summation equation. To cleanly separate the index math, we split the sum over $c$ into the single matching term ($c = i$) and all non-matching terms ($c \neq i$):

$$\frac{\partial \mathcal{L}}{\partial z_i} = \left( \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_i} \right) + \sum_{c \neq i}^C \left( \frac{\partial \mathcal{L}}{\partial \hat{y}_c} \frac{\partial \hat{y}_c}{\partial z_i} \right)$$

Substitute the explicit derivatives we derived in Steps A and B:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \left( -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i(1 - \hat{y}_i) \right) + \sum_{c \neq i}^C \left( -\frac{y_c}{\hat{y}_c} \cdot (-\hat{y}_c \hat{y}_i) \right)$$

Simplify by canceling out the denominators ($\hat{y}_i$ and $\hat{y}_c$):
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i(1 - \hat{y}_i) + \sum_{c \neq i}^C y_c \hat{y}_i$$
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + y_i \hat{y}_i + \sum_{c \neq i}^C y_c \hat{y}_i$$

Notice that $\hat{y}_i$ can be factored out of both terms on the right side:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + \hat{y}_i \left( y_i + \sum_{c \neq i}^C y_c \right)$$
$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + \hat{y}_i \left( \sum_{c=1}^C y_c \right)$$

Because $\mathbf{y}$ is a true probability distribution (one-hot encoded), the sum of all elements $\sum_{c=1}^C y_c = 1$. This reduces the equation to:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

Converting this index notation back into full matrix-vector formatting for our batch operations yields the vector difference:
$$\mathbf{\delta}_{out} = \mathbf{\hat{y}} - \mathbf{y}$$

### 4. Implementation
```python
import numpy as np

class MulticlassNeuralNetwork:
    def __init__(self, layers):
        """
        layers: List containing the number of neurons in each layer.
                e.g., [4, 8, 3] means 4 inputs, 8 hidden units, 3 output classes.
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights using He (Kaiming) initialization for ReLU
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, a):
        # Expects 'a' to be the activated output: relu(z)
        # Gradient is 1 where z > 0 (which maps to a > 0), and 0 otherwise
        return (a > 0).astype(float)

    def _softmax(self, z):
        # Subtract max(z) along the row axis for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network using ReLU for hidden layers
        and Softmax for the final classification output layer.
        """
        self.as_ = [X]
        self.zs = []
        
        a = X
        # Pass through hidden layers with ReLU
        for l in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[l]) + self.biases[l]
            a = self._relu(z)
            self.zs.append(z)
            self.as_.append(a)
            
        # Final layer with Softmax classification
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self._softmax(z_out)
        self.zs.append(z_out)
        self.as_.append(a_out)
            
        return a_out

    def backward(self, X, y, learning_rate):
        """
        Backward pass combining Categorical Cross-Entropy Loss with Softmax,
        propagating errors backward using ReLU derivatives.
        """
        m = X.shape[0] # Batch size
        
        # 1. Delta for Softmax + Cross-Entropy Loss simplifies beautifully to: (y_pred - y)
        y_pred = self.as_[-1]
        delta = (y_pred - y)
        
        # Iterate backwards through the layers
        for l in reversed(range(len(self.weights))):
            # Calculate gradients for parameters at layer l
            grad_w = np.dot(self.as_[l].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            # If we aren't at the first layer, propagate delta backward using the ReLU derivative
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self._relu_derivative(self.as_[l])
            
            # Update parameters via Gradient Descent
            self.weights[l] -= learning_rate * grad_w
            self.biases[l] -= learning_rate * grad_b

    def train(self, X, y, epochs, learning_rate):
        """
        y must be provided as a one-hot encoded matrix.
        """
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 200 == 0:
                # Calculate Categorical Cross-Entropy Loss
                # Clip values to avoid log(0)
                y_pred = np.clip(self.as_[-1], 1e-15, 1.0 - 1e-15)
                loss = -np.sum(y * np.log(y_pred)) / X.shape[0]
                
                # Calculate classification accuracy
                acc = np.mean(np.argmax(self.as_[-1], axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")


# --- Verification on a 3-Class Synthetic Dataset ---
if __name__ == "__main__":
    # Create a small classification problem: 4 samples, 4 features, 3 target classes
    X = np.array([
        [0.1, 0.9, 0.2, 0.4],
        [0.8, 0.1, 0.7, 0.9],
        [0.2, 0.3, 0.9, 0.1],
        [0.6, 0.7, 0.1, 0.2]
    ])
    
    # Target class labels: Class 0, Class 2, Class 1, Class 0
    y_labels = np.array([0, 2, 1, 0])
    
    # Convert labels to One-Hot Encoding
    num_classes = 3
    y_one_hot = np.eye(num_classes)[y_labels]

    # Initialize network: 4 inputs -> 6 hidden units -> 3 output probabilities
    nn = MulticlassNeuralNetwork(layers=[4, 6, num_classes])
    
    print("Training Multiclass MLP (ReLU + Softmax)...")
    nn.train(X, y_one_hot, epochs=2001, learning_rate=0.1)
    
    # Test predictions after training
    prob_predictions = nn.forward(X)
    class_predictions = np.argmax(prob_predictions, axis=1)
    
    print("\nFinal Results:")
    for i, (prob, pred, target) in enumerate(zip(prob_predictions, class_predictions, y_labels)):
        formatted_probs = [f"{p:.4f}" for p in prob]
        print(f"Sample {i} | Probabilities: {formatted_probs} | Predicted Class: {pred} (Target: {target})")

```
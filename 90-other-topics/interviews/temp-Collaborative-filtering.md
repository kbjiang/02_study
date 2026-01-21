# Matrix Factorization – Numerical Example

## Step 1: Rating Matrix

Suppose we have 3 users and 3 movies:

$$  
R =  
\begin{bmatrix}  
5 & 3 & ? \\  
4 & ? & 1 \\  
? & 2 & 5  
\end{bmatrix}  
$$

`?` = missing ratings we want to predict.

---

## Step 2: Choose Latent Dimension

Let:

- $k = 2$ latent factors
    

We approximate:

$$  
R \approx U V^T  
$$

Where:

- $U \in \mathbb{R}^{3 \times 2}$ (users)
    
- $V \in \mathbb{R}^{3 \times 2}$ (items)
    

---

## Step 3: Learned Embeddings

Assume after training:

$$  
U =  
\begin{bmatrix}  
1.2 & 0.5 \  
1.0 & 0.3 \  
0.2 & 1.1  
\end{bmatrix}  
$$

$$  
V =  
\begin{bmatrix}  
1.5 & 0.4 \  
0.9 & 0.1 \  
0.1 & 1.3  
\end{bmatrix}  
$$

---

## Step 4: Predict Ratings

Prediction rule:

$$  
\hat r_{ui} = u_u^T v_i  
$$

### Example 1: Predict (User 1, Movie 3)

$$  
u_1 = [1.2, 0.5]  
$$

$$  
v_3 = [0.1, 1.3]  
$$

$$  
\hat r_{13}  
= (1.2)(0.1) + (0.5)(1.3)  
$$

$$  
= 0.12 + 0.65 = 0.77  
$$

**Predicted rating ≈ 0.77**

---

### Example 2: Predict (User 2, Movie 2)

$$  
u_2 = [1.0, 0.3]  
$$

$$  
v_2 = [0.9, 0.1]  
$$

$$  
\hat r_{22}  
= (1.0)(0.9) + (0.3)(0.1)  
$$

$$  
= 0.9 + 0.03 = 0.93  
$$

**Predicted rating ≈ 0.93**

---

## Step 5: Reconstructed Matrix

$$  
\hat R = U V^T =  
\begin{bmatrix}  
2.0 & 1.1 & 0.77 \\  
1.62 & 0.93 & 0.49 \\  
0.74 & 0.29 & 1.45  
\end{bmatrix}  
$$

(rounded values)

---

## Step 6: Training Objective

We learned $U$ and $V$ by minimizing:

$$  
\sum_{(u,i)\in obs}  
(r_{ui} - u_u^T v_i)^2

- \lambda(|u_u|^2 + |v_i|^2)  
    $$
    

Optimized using:

- Stochastic Gradient Descent (SGD)
    
- Alternating Least Squares (ALS)
    

---

## Step 7: Intuition

- Each user & item lives in a **latent space**
    
- Prediction = **vector alignment**
    
- Similar directions → high rating
    
- Orthogonal → low rating
    

---

## Key Takeaways

- Handles sparse data
    
- Learns hidden preference dimensions
    
- Predictions via dot products
    
- Scales to large systems
## 1. What is regression to the mean?

**Regression to the mean** means:

> If something is *extreme* on one measurement,  
> it tends to be *less extreme* the next time.

Examples:
- Someone scores **very high** on a test → next test likely lower  
- Someone has a **terrible day** → next day probably better  
- An athlete has an **amazing game** → next game usually more normal  

Important:
- This happens **even if nothing real changes**
- It is a **statistical effect**, not psychological

---

## 2. Intuition

Any measurement can be thought of as:

$$
true signal + random noise
$$

Extreme outcomes usually happen because:
- The signal is high (or low) **and**
- Random noise pushes it even further

Next time:
- Noise is random again  
- The lucky/unlucky push disappears  

➡ The result moves **closer to the average**.

---

## 3. Role of correlation

Let:
- X = first measurement  
- Y = second measurement  
- ρ = Corr(X,Y)  

Then:

$$
E[Y | X=x]
= μ_Y + ρ (σ_Y / σ_X)(x - μ_X)
$$

Key term:

$$
ρ(x - μ_X)
$$

This controls how much the second measurement depends on the first.

---

## 4. Special cases

### Perfect correlation (ρ = 1)

$$
Y = X
$$

- No regression to the mean  
- Extreme stays extreme  

---

### Zero correlation (ρ = 0)

$$
E[Y|X=x] = μ_Y
$$

- Complete regression  
- Next value = average  
- Past tells you nothing  

---

### Partial correlation (0 < ρ < 1)

- Some regression  
- Extreme values shrink toward the mean  

---

## 5. Geometric intuition

Think of a scatterplot:

- Perfect diagonal line → no regression  
- Cloud of points → imperfect correlation  
- Best-fit line has slope < 1  
- Extreme X → predicted Y less extreme  

That slope is related to **correlation**.

---

## 6. Common mistake

People often think:

"They did worse because they relaxed."  
"They improved because of coaching."

But often:

It’s just **regression to the mean**.

---

## 7. One-sentence summary

**Regression to the mean happens because correlation is less than 1.**  

Random noise cancels out over time, pulling extreme observations back toward a

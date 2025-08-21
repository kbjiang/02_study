## Running example
**Given a dataset [3, 5, 13, 9, 8], how do we calculate the 95% confidence interval?**

---
## 1. Key Concepts involved

### Sample Standard Deviation
- **Substitution Principle:** Sample std (s) substitutes for unknown population std (σ)
### Standard Error
- Standard Error measures **how much sample means would vary** if you took many samples from the same population.
- SE = s/√m (assuming σ is not available), where m is the sample size
---
## 2. Sample Standard Deviation vs Population Standard Deviation

### The Core Difference: (n-1) vs n
**Population Standard Deviation:**
- Used when you have data for the entire population
- Formula: σ = √[Σ(x - μ)² / n]
- Divides by n (total number of data points)

**Sample Standard Deviation:**
- Used when you have a sample from a larger population (most common case)
- Formula: s = √[Σ(x - x̄)² / (n-1)]
- Divides by (n-1) - this is called **Bessel's correction**

### Why (n-1)? The Degrees of Freedom Connection
**Intuitive Explanation:**
- Once you calculate the sample mean, you lose one degree of freedom
- Dividing by (n-1) makes the sample standard deviation an **unbiased estimator** of population standard deviation

### Numerical Example
- Mean = (3 + 5 + 13 + 9 + 8) ÷ 5 = 7.6
- Sum of deviations = (3 - 7.6)² + (5 - 7.6)² + (13 - 7.6)² + (9 - 7.6)² + (8 - 7.6)² = 59.2
- Sample variance = 59.2 ÷ (5-1) = 59.2 ÷ 4 = 14.8
- Sample std = √14.8 = 3.847
- Population variance = 59.2 ÷ 5 = 11.84
- Population std = √11.84 = 3.441
---
## 3. Standard Error: From Sample Std to Precision of the Mean
1. **Theoretical Foundation:**
   - If you took many samples of size n, each would have a slightly different mean
   - Standard Error = standard deviation of these sample means
   - From probability theory: SE = σ/√n (true formula)

2. **Practical Application:**
   - We don't know σ (population std), so we substitute s (sample std)
   - Therefore: SE = s/√n

3. **Why √n in denominator?**
   - Comes from variance properties of independent random variables
   - Variance of sample mean = σ²/n
   - Standard Error = √(σ²/n) = σ/√n

### Numerical Example
- Sample std (s) = 3.847
- Sample size (n) = 5
- Standard Error = 3.847 ÷ √5 = 3.847 ÷ 2.236 = **1.720**
---
## 4. Building the 95% Confidence Interval
**Step 1: Sample Statistics**
- Sample mean (x̄) = 7.6
- Sample std (s) = 3.847
- Sample size (n) = 5

**Step 2: Degrees of Freedom**
- df = n - 1 = 4

**Step 3: Standard Error**
- SE = s/√n = 3.847/√5 = 1.720

**Step 4: Critical t-value**
- For 95% confidence with df=4: t₀.₀₂₅,₄ = 2.776
- (We use t-distribution because sample size is small)

**Step 5: Margin of Error**
- Margin of Error = t × SE = 2.776 × 1.720 = 4.775

**Step 6: Confidence Interval**
- CI = x̄ ± Margin of Error
- CI = 7.6 ± 4.775
- **95% CI: [2.825, 12.375]**

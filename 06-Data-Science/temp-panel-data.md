### Q: How is the FE connected with parallel trends?
 Great question, Simon. They're deeply connected — parallel trends is the **identifying assumption** that makes FE a valid causal estimator.

**The link:**
FE removes time-invariant differences between entities (different intercepts). What remains is each entity's **trajectory over time**. The causal estimate comes from comparing how trajectories change when treatment switches on vs. off.
For this to work, you need: **in the absence of treatment, entities would have followed the same trajectory** (i.e., parallel trends).

**Step by step:**
1. FE demeans by entity → removes level differences (Person A always earns more than Person B)
2. After demeaning, you're looking at **changes** within each person
3. If person A gets married in year 5 and their wage jumps, is that causal?
4. Only if their wage **would have followed the same trend** as unmarried people's wages, had they not married
5. That's parallel trends: $\big(Y_d(t) - Y_d(t-1)\big) \perp D$

**If parallel trends is violated:**
Say people who get married are also on steeper career trajectories (they were going to get raises anyway, marriage or not). Then:
- FE removes their higher *level* (good)
- But their steeper *growth* still confounds the estimate (bad)
- The marriage coefficient picks up both the true effect + the differential growth → biased

**Think of it as two layers of confounding:**

| Confounding type                         | What handles it                            |
| ---------------------------------------- | ------------------------------------------ |
| Level differences (time-invariant)       | Entity FE removes them                     |
| Growth differences (time-varying trends) | Parallel trends *assumes* they don't exist |
FE is the **method**. Parallel trends is the **assumption** under which the method gives you a causal answer. Without parallel trends, FE still removes level differences, but the remaining estimate is biased by differential trends.

This is also why adding **Time FE** helps: it absorbs common time shocks (inflation, recessions). But even with both Entity + Time FE, you still need parallel trends for the entity-specific deviations from the common trend.


---
---
### Q: Least Squares / Two‑Way ANOVA and Its Connection to Panel Data
#### Problem Setup
- **Entities (Pages)**: $j = 1,\dots,1000$
- **Groups (Reviewers)**: $i = 1,\dots,50$
- **Observations**: Each reviewer rates 100 pages → **5000 total ratings**
- **Outcome**: $y_{ij} \in [0,10]$ (usability rating)
---
#### Two‑Way ANOVA (Least Squares) Model
$$
y_{ij} = \mu + \alpha_j + \beta_i + \varepsilon_{ij}
$$
- $\mu$: global mean usability 
- $\alpha_j$: **page effect** (intrinsic usability of page $j$) 
- $\beta_i$: **reviewer effect** (harshness / leniency bias) 
- $\varepsilon_{ij}$: noise with mean 0 
**Identifiability constraints**
$$

\sum_j \alpha_j = 0,
\qquad
\sum_i \beta_i = 0
$$
Interpretation: page and reviewer effects are deviations from the average.
#### Estimation via Least Squares / ANOVA
- Stack the 5000 observed ratings into a vector $y$.
- Regress on **page dummies + reviewer dummies + intercept**.
- Solve with ordinary least squares (OLS).
This is exactly a **two‑way ANOVA without interaction**.
---
#### Closed‑Form Intuition
Let:
- $\bar y_{j\cdot}$ = mean rating of page $j$ 
- $\bar y_{\cdot i}$ = mean rating by reviewer $i$ 
- $\bar y_{\cdot\cdot}$ = global mean 
Then the OLS solution is:
$$
\hat\alpha_j = \bar y_{j\cdot} - \bar y_{\cdot\cdot}
$$
$$
\hat\beta_i = \bar y_{\cdot i} - \bar y_{\cdot\cdot}
$$
$$
\hat\mu = \bar y_{\cdot\cdot}
$$
---
#### Usability Score
For page $j$:
$$
\text{UsabilityScore}_j = \hat\mu + \hat\alpha_j
$$
This is the **reviewer‑adjusted page quality**.

---
#### Data Matrix vs Design Matrix
- **Ratings matrix $Y$**:
  - Shape: **1000 × 50**
  - Sparse / missing (only 5000 observed out of 50,000 possible)
- **Design matrix $X$** (regression form):
  - Rows: **5000** (one per observed rating)
  - Columns: **1000 page dummies + 50 reviewer dummies (+ intercept)**
  - Extremely sparse: each row has only 2–3 non‑zero entries
---
#### Connection to Panel Data
This model **is a two‑way fixed‑effects (TWFE) panel model**.

| ANOVA / Stats view        | Panel / Econometrics view |
| ------------------------- | ------------------------- |
| Page effect $\alpha_j$    | Entity fixed effect       |
| Reviewer effect $\beta_i$ | Group / time fixed effect |
| Dummy‑variable OLS        | Fixed‑effects estimator   |
| Two‑way ANOVA             | Two‑way fixed effects     |
Equivalent **within (double‑demeaning) transformation**:
$$
y_{ij}
- \bar y_{j\cdot}
- \bar y_{\cdot i}
+ \bar y_{\cdot\cdot}
$$
---
---
### Follow-up: why no interaction term $\alpha_i*\beta_j$ in this Two‑Way ANOVA / Fixed‑Effects Model
#### What an Interaction Term Means
Baseline (additive) model:
$$
y_{ij} = \mu + \alpha_j + \beta_i + \varepsilon_{ij}
$$
- $\alpha_j$: page quality (constant across reviewers)
- $\beta_i$: reviewer bias (constant across pages)
Adding an interaction:
$$
y_{ij} = \mu + \alpha_j + \beta_i + \gamma_{ij} + \varepsilon_{ij}
$$
- $\gamma_{ij}$ captures **reviewer‑specific behavior on a specific page**. E.g., “Reviewer 7 is unusually harsh on Page 312”).
---
#### Parameter Counting (Why This Breaks)
- Pages: 1000 → $\alpha_j$ (999 df)
- Reviewers: 50 → $\beta_i$ (49 df)
- Interactions: $1000 \times 50 = 50{,}000$ possible $\gamma_{ij}$
- Observations: **5000 total ratings**
You would be estimating **far more parameters than data**.
---
#### Intuition Using One Page
- Each page has **5 ratings**
- With interaction, each rating effectively gets its **own parameter**
- No replication remains to estimate noise
Result:
- Residual variance = 0
- Perfect fit, zero generalization
This is what people mean by:
> “You’d have no degrees of freedom left (5 ratings per page).”

---
#### Panel‑Data Interpretation
In panel terms:
$$
y_{ij} = \alpha_j + \beta_i + \gamma_{ij} + \varepsilon_{ij}
$$
- $\alpha_j$: entity fixed effects
- $\beta_i$: group/time fixed effects
- $\gamma_{ij}$: **entity × time fixed effects**
This is equivalent to:
- One dummy per observation
- A saturated model
✅ Fits the data exactly 
❌ No estimable error variance 
❌ No inferential value

---
#### When Interaction *Would* Make Sense
Only if you had **replication per (reviewer, page) cell**, e.g.:
- Multiple ratings per reviewer–page pair
- Multiple reviewers of the same “type” per page
Then interaction effects become estimable.

---
#### **One‑liner** 
> Adding a reviewer × page interaction gives each observed rating its own parameter; with only five ratings per page and no replication per cell, the interaction absorbs all variation and leaves zero degrees of freedom for error, making the model unidentifiable.
  

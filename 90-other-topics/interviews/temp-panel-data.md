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
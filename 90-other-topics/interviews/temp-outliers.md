### Table 1 — Outlier Detection Methods
| Method Type        | Techniques                             | When to Use                                                   |
| ------------------ | -------------------------------------- | ------------------------------------------------------------- |
| **Statistical**    | Z‑score, Modified Z‑score, IQR rule    | Data roughly unimodal; numeric features; simple distributions |
| **Model‑based**    | Isolation Forest, One‑Class SVM, LOF   | Complex distributions; high‑dimensional data                  |
| **Distance‑based** | K‑NN distance, DBSCAN                  | Clustering tasks; spatial/anomaly detection                   |
| **Domain‑based**   | Business rules, operational thresholds | When domain knowledge defines “impossible” values             |
| **Residual‑based** | Analyze model residuals                | Regression tasks to identify poorly explained points          |
### Table 2 — Outlier Handling Strategies
| Strategy              | Action                                              | When Appropriate                                               |
| --------------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| **Remove**            | ==Drop clear errors or impossible values==          | Data errors, sensor issues, corrupt logs                       |
| **Cap/Winsorize**     | ==Replace extreme tails with chosen percentiles==   | Heavy‑tailed but real data; want stability in linear models    |
| **Transform**         | Log, sqrt, Box‑Cox                                  | Right‑skewed distributions; reduce influence of extremes       |
| **Impute**            | Replace outlier with median or model‑based estimate | When removing would remove too much data                       |
| **Use robust models** | Trees, Huber regression, quantile models            | When outliers are real but should not dominate                 |
| **Keep**              | Leave outliers untouched                            | Outliers are meaningful (fraud, churn, rare high‑value events) |

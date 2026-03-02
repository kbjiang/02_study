## `reset_index` vs `unstack` vs `pivot_table`

| | `reset_index()` | `unstack()` | `pivot_table()` |
|---|---|---|---|
| **What it does** | Moves index levels into regular columns | Pivots an index level into column headers (long → wide) | Groups, aggregates, and pivots in one step (long → wide) |
| **Input** | Any DataFrame (usually with MultiIndex) | Already grouped/MultiIndexed data | Raw (ungrouped) data |
| **Aggregation** | No | No — just reshapes | Yes — applies `aggfunc` (mean, sum, etc.) |
| **Shape change** | Same rows, more columns | Fewer rows, wider | Fewer rows, wider |
| **NaN risk** | No | Yes (missing combinations) | Yes (missing combinations) |
| **Typical use** | Flatten a MultiIndex back to a regular table | After `.groupby().agg()` to reshape | Directly on raw DataFrame for summary |
### Sample Data
```python
import pandas as pd

df = pd.DataFrame({
    'region':  ['East', 'East', 'East', 'West', 'West'],
    'product': ['A',    'A',    'B',    'A',    'A'],
    'sales':   [10,     15,     20,     30,     25]
})

#   region product  sales
# 0 East   A        10
# 1 East   A        15
# 2 East   B        20
# 3 West   A        30
# 4 West   A        25

df.pivot_table(values='sales', index='region', columns='product', aggfunc='sum')
# product    A      B
# region
# East      25    20.0
# West      55     NaN

grouped = df.groupby(['region', 'product'])['sales'].sum()
#              sales
# region product
# East   A        25
#        B        20
# West   A        55

grouped.unstack('product') # identical to `pivot_table` result
# product    A      B
# region
# East      25    20.0
# West      55     NaN

grouped.reset_index()
#   region product  sales
# 0 East   A        25
# 1 East   B        20
# 2 West   A        55
```


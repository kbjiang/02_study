### The problem:
Entity FE controls for things that differ across *people* but stay constant over *time* (beauty, intelligence). But what about the reverse — things that change over *time* but are the same for *everyone* in a given year?

Example: Inflation. In 1985, everyone's salary is higher than in 1980, regardless of who they are. If both marriage rates AND wages are trending upward over time:
- More people get married in later years
- Everyone earns more in later years (inflation)
- This creates a spurious positive correlation between marriage and wages — not because marriage causes higher wages, but because **both happen to increase with time**
Time itself is a confounder here.
### The solution:
Add a dummy for each year — just like we added a dummy for each person. This absorbs any year-specific shocks that affect everyone equally:
- Inflation
- Recessions
- Policy changes
- Cultural trends

### Symmetry with entity FE:

|                             | Entity FE                                | Time FE                                    |
| --------------------------- | ---------------------------------------- | ------------------------------------------ |
| Dummy for each...           | person                                   | year                                       |
| Controls for things that... | differ across people, constant over time | differ across time, constant across people |
| Demeaning equivalent        | subtract person mean                     | subtract year mean                         |
### Why the coefficient dropped from 0.1147 to 0.0476:
Part of the apparent marriage → wage effect was just "both marriage and wages go up over time." Once you absorb those time trends with year dummies, the remaining causal effect is smaller but still significant.

### **In FWL terms:** 
$X_2$ now includes BOTH person dummies AND year dummies. Partialling out both means you're looking at variation that's neither explained by "which person" nor by "which year" — only the truly idiosyncratic within-person, within-year variation remains.
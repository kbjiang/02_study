# TF vs TF-IDF vs Normalized TF

This document explains the differences between **TF**, **normalized TF**, and **TF-IDF** using a concrete example.

---

## Example sentence
> **“the cat chased the cat”**

Vocabulary:
```
[the, cat, chased]
```

Raw counts:
```
the: 2
cat: 2
chased: 1
```

---

## 1. TF (Term Frequency)

### Raw TF
```
[the:2, cat:2, chased:1]
```

### Length-normalized TF (optional)
Total words = 5
```
[the:0.4, cat:0.4, chased:0.2]
```

**Limitations**
- Common words dominate
- Longer documents dominate shorter ones

---

## 2. Normalized TF

Take raw TF and normalize the vector.

### L2-normalized TF
Vector:
```
[2, 2, 1]
```

L2 norm:
```
√(2² + 2² + 1²) = 3
```

Normalized TF:
```
[0.667, 0.667, 0.333]
```

**What this fixes**
- Document length bias
- Enables cosine similarity

**What it does not fix**
- Common words are still overweighted

---

## 3. TF-IDF

TF-IDF penalizes words that appear in many documents.

### IDF formula
```
IDF(t) = log(N / df(t))
```

Assume corpus statistics:
```
N = 1000
df(the) = 900
df(cat) = 10
df(chased) = 50
```

IDF values:
```
the     → 0.11
cat     → 4.61
chased  → 3.00
```

### TF-IDF (raw)
```
the:     0.22
cat:     9.22
chased:  3.00
```

---

## 4. Normalized TF-IDF

TF-IDF vectors are almost always L2-normalized.

Approximate normalized TF-IDF:
```
[0.022, 0.94, 0.31]
```

This is the **standard classical NLP representation**.

---

## Summary Table

| Representation | Normalized | Penalizes common words |
|---------------|------------|-----------------------|
| TF (raw) | No | No |
| Normalized TF | Yes | No |
| TF-IDF | No | Yes |
| Normalized TF-IDF | Yes | Yes |

---

## Intuition

- **TF**: How often a word appears in this document
- **Normalized TF**: Importance within this document
- **TF-IDF**: Importance relative to the entire corpus

---

## Why this still matters

- Strong baseline in NLP
- Works well with cosine similarity and linear models
- Conceptual ancestor of embedding pooling

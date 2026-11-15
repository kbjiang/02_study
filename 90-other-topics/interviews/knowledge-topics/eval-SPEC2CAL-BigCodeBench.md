# How Should an LLM Judge Assess Code Correctness Without Runtime Execution?

The judge should reason about how the code would behave according to the task specification, but it should not pretend that it has verified runtime behavior when it hasn't executed the code.

## 1. Clear violation — should reject

**Task:** Return the square of a number.

```python
def square(x):
    return x + x
```

The judge can confidently identify the incorrect computation from the source code. No execution is necessary.

## 2. Reasonable behavioral inference — can reject

**Task:** Return 0 when the input list is empty.

```python
def average(numbers):
    return sum(numbers) / len(numbers)
```

The judge can reason that an empty list causes division by zero. This is not arbitrary speculation; it's a direct consequence of the code and the stated requirement.

## 3. Uncertain runtime behavior — should not reject speculatively

**Task:** Parse a date string and return a datetime object.

```python
from dateutil.parser import parse

def parse_date(value):
    return parse(value)
```

Without knowing the runtime environment, installed dependencies, and actual input, the judge shouldn't invent a failure and mark the code incorrect.

It may identify a potential dependency or edge-case concern, but that is not sufficient evidence of functional incorrectness.

## 4. Appropriate vs. inappropriate judge behavior

| Judge behavior | Appropriate? |
|---|---|
| Identify a clear logical error from source code | Yes |
| Reason about an evident consequence of the code | Yes |
| Identify a violation of the task specification | Yes |
| Invent hypothetical runtime failures without evidence | No |
| Reject code simply because it looks unusual or stylistically poor | No |
| Claim execution has verified correctness when it hasn't | No |

## 5. Interview-ready explanation

> Since the judge doesn't execute the candidate code, it should assess functional correctness through static reasoning against the task specification. It can identify clear logical errors and reason about evident runtime consequences, but it should require concrete evidence before assigning a FAIL. It shouldn't reject code based on speculative runtime assumptions or stylistic preferences.

## Key takeaway

**The goal is not to judge how the code looks; it's to judge the behavior that can reasonably be inferred from the code.**

That distinction is central to the BigCodeBench rubric revision.

# Math Algorithms

Some algorithms I made to understand maths better.

## Binomial law

```python
from binomial_law import *

# Binomial law of parameters n = 10 and p = 0.2
X = Binomial(10, 0.2)

print(f"X -> {X}")
print(f"Expectation of X: {X.expected}")
print(f"Variance of X:  {X.variance}")
print(f"Sigma of X:  {X.sigma}")
print(f"P(X=8)  = {X == 8}")
print(f"P(X>8)  = {X > 8}")
print(f"P(X>=8) = {X >= 8}")
print(f"P(X<8)  = {X < 8}")
print(f"P(X<=8) = {X <= 8}")
```

Also provide function to visualize a law using `matplotlib.pyplot`:

```python
from binomial_law import *

X = Binomial(10, 0.2)

show_graph(X)
```

![PyCharm](images/binomial_law.png)


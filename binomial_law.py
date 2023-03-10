import functools
import math

import matplotlib.pyplot as plt


@functools.lru_cache(maxsize=None)
def _m(m: int) -> int:
	"""
	>>> _m(0)
	1
	"""
	return m == 0 and 1 or math.prod(range(1, m + 1))


@functools.lru_cache(maxsize=None)
def coef(n: int, k: int) -> float:
	"""
	>>> coef(7, 3)
	35.0

	>>> coef(7, 4)
	35.0

	>>> coef(10, 2)
	45.0

	>>> coef(10, 8)
	45.0

	>>> coef(0, 0)
	1.0

	>>> coef(1, 0)
	1.0
	"""
	return _m(n) / (_m(k) * _m(n - k))


class Binomial:
	"""
	Represents a Binomial distribution of parameter n and p
	"""

	def __init__(self, n: int, p: float):
		self.n = n
		self.p = p

	def __str__(self):
		"""
		>>> str(Binomial(10, 20))
		'B(10, 20)'
		"""
		return f"B({self.n}, {self.p})"

	@functools.cached_property
	def expected(self) -> float:
		"""
		E(x) = np

		>>> Binomial(10, 20).expected
		200
		"""
		return self.n * self.p

	@functools.cached_property
	def variance(self) -> float:
		"""
		V(X) = np(1 - p)
		"""
		return self.expected * (1 - self.p)

	@functools.cached_property
	def sigma(self) -> float:
		"""
		sigma(X) = sqrt(np(1 - p))
		"""
		return math.sqrt(self.variance)

	def __eq__(self, k: int) -> float:
		"""
		P(X=k)

		>>> Binomial(20, 0.45) == 8
		0.16230037134255265
		"""
		return coef(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

	def __gt__(self, k: int) -> float:
		"""
		P(X>k)

		>>> Binomial(20, 0.45) > 8
		0.585693765883885
		"""
		p = 0
		for i in range(self.n, k, -1):
			p += self == i
		return p

	def __ge__(self, k: int) -> float:
		"""
		P(X>=k)

		>>> Binomial(20, 0.45) >= 8
		0.7479941372264376
		"""
		p = 0
		for i in range(self.n, k - 1, -1):
			p += self == i
		return p

	def __lt__(self, k: int) -> float:
		"""
		P(X<k)

		>>> Binomial(20, 0.45) < 8
		0.25200586277356357
		"""
		p = 0
		for i in range(0, k):
			p += self == i
		return p

	def __le__(self, k: int) -> float:
		"""
		P(X<=k)

		>>> Binomial(20, 0.45) <= 8
		0.4143062341161162
		"""
		p = 0
		for i in range(0, k + 1):
			p += self == i
		return p


def show_graph(b: Binomial):
	# plt.axis((0, b.n, 0, None))
	for x in range(0, b.n):
		plt.bar(x, b == x, width=0.4, color="orange")
	plt.grid()
	plt.show()


if __name__ == "__main__":
	X = Binomial(20, 0.45)
	# X = Binomiale(50, 0.17)

	print(f"X -> {X}")
	print(f"Expectation of X: {X.expected}")
	print(f"Variance of X:  {X.variance}")
	print(f"Sigma of X:  {X.sigma}")
	print(f"P(X=8)  = {X == 8}")
	print(f"P(X>8)  = {X > 8}")
	print(f"P(X>=8) = {X >= 8}")
	print(f"P(X<8)  = {X < 8}")
	print(f"P(X<=8) = {X <= 8}")

	show_graph(X)

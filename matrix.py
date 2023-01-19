from __future__ import annotations

from typing import List, Tuple, NamedTuple, Generator, Any


class _MatrixSize(NamedTuple):
	"""
	Represent the size of a matrix
	"""
	n: int
	p: int

	def __eq__(self, other: object) -> bool:
		if isinstance(other, _MatrixSize):
			return self.n == other.n and self.p == other.p
		elif isinstance(other, tuple):
			return self.n == other[0] and self.p == other[1]
		else:
			raise NotImplementedError

	def __str__(self) -> str:
		return f"(n={self.n}, p={self.p})"


class _MatrixIndex(NamedTuple):
	"""
	Represent an index of a matrix
	"""
	i: int
	j: int

	def __eq__(self, other: object) -> bool:
		if isinstance(other, _MatrixIndex):
			return self.i == other.i and self.j == other.j
		elif isinstance(other, tuple):
			return self.i == other[0] and self.j == other[1]
		else:
			raise NotImplementedError

	def __str__(self) -> str:
		return f"(i={self.i}, j={self.j})"


_MatrixData = List[List[int | float]]


class Matrix:
	"""
	Represent a Matrix

	A matrix is considered immutable
	"""

	def __init__(self, data: _MatrixData):
		self.data = data

	def __getitem__(self, params: Tuple[int, int]) -> float:
		return self.index(params[0], params[1])  # i, j

	def __str__(self) -> str:
		return self.ascii_format()

	def __eq__(self, other: object) -> bool:
		if isinstance(other, Matrix):
			return self.data == other.data
		else:
			raise NotImplementedError

	def __add__(self, other: object) -> Matrix:
		if isinstance(other, Matrix):
			assert self.size == other.size
			data: _MatrixData = self.__empty_data(self.size.n, self.size.p)
			for idx in self.iter():
				data[idx.i - 1][idx.j - 1] = self.index(idx.i, idx.j) + other.index(idx.i, idx.j)
			return Matrix(data)
		else:
			raise NotImplementedError

	def __neg__(self):
		"""
		:return: The opposite matrix (a_ij = -a_ij)
		"""
		data = self.__empty_data(self.size.n, self.size.p)

		for idx in self.iter():
			data[idx.i - 1][idx.j - 1] = -self.index(idx.i, idx.j)

		return Matrix(data)

	def __mul__(self, other: object) -> Matrix:
		if isinstance(other, (float, int)):
			data: _MatrixData = self.__empty_data(self.size.n, self.size.p)
			for idx in self.iter():
				data[idx.i - 1][idx.j - 1] = self.index(idx.i, idx.j) * other
			return Matrix(data)

		elif isinstance(other, Matrix):
			size_1 = self.size  # n, p
			size_2 = other.size  # p, q

			assert size_1.p == size_2.n  # check condition

			p = size_1.p

			# Calculate the size of the resulting matrix (n, q)
			# Create empty matrix data
			data = self.__empty_data(size_1.n, size_2.p)

			for i in range(0, size_1.n):
				for j in range(0, size_2.p):
					data[i][j] = sum([self.index(i + 1, k) * other.index(k, j + 1) for k in range(1, p + 1)])

			return Matrix(data)
		else:
			raise NotImplementedError

	def copy(self) -> Matrix:
		"""
		:return: Return a copy of the matrix
		"""
		return Matrix(self.data)

	@property
	def size(self) -> _MatrixSize:
		"""
		:return: The size of the matrix (n, p)
		"""
		return _MatrixSize(len(self.data), len(self.data[0]))

	def index(self, i: int, j: int) -> int | float:
		"""
		:param i: Index i, line
		:param j: Index j, column
		:return: The number m_ij
		"""
		assert i >= 1 and j >= 1
		return self.data[i - 1][j - 1]

	def iter(self) -> Generator[_MatrixIndex, Any, None]:
		"""
		Iterate on each index of a matrix (by line)
		"""
		for i in range(1, self.size.n + 1):
			for j in range(1, self.size.p + 1):
				yield _MatrixIndex(i, j)

	def ascii_format(self) -> str:
		"""
		:return: WIP ascii representation of the matrix
		"""
		return "\n".join([f"|{' '.join([str(self.index(i, j)) for j in range(1, self.size.p + 1)])}|" for i in
		                  range(1, self.size.n + 1)])

	@property
	def is_line(self) -> bool:
		"""
		:return: True if matrix is a line matrix (n=1)
		"""
		return self.size.n == 1

	@property
	def is_column(self) -> bool:
		"""
		:return: True if matrix is a column matrix (p=1)
		"""
		return self.size.p == 1

	@property
	def is_square(self) -> bool:
		"""
		:return: True if matrix is a square matrix (n=p)
		"""
		return self.size.n == self.size.p

	@property
	def order(self) -> int:
		"""
		:return: The order of a square matrix
		"""
		assert self.is_square
		return self.size.n

	@property
	def diagonal(self) -> List[float]:
		"""
		:return: The diagonal of a square matrix, ie the list of coefs a_11, a_22, ..., a_nn
		"""
		return [self.index(i, i) for i in range(1, self.order + 1)]

	# Specific constructors

	@classmethod
	def __empty_data(cls, n: int, p: int) -> _MatrixData:
		return [[0 for _ in range(p)] for _ in range(n)]

	@classmethod
	def empty(cls, n: int, p: int) -> Matrix:
		"""
		:return: An empty matrix of size (n, p)
		"""
		return Matrix(cls.__empty_data(n, p))

	@classmethod
	def null(cls, order: int) -> Matrix:
		"""
		:param order: The order of the matrix
		:return: The null matrix
		"""
		return cls.empty(order, order)

	@classmethod
	def identity(cls, order: int) -> Matrix:
		"""
		:param order: The order of the matrix
		:return: The identity matrix of given order
		"""
		data: _MatrixData = [[0 for _ in range(order)] for _ in range(order)]
		for i in range(order):
			data[i][i] = 1
		return Matrix(data)


if __name__ == "__main__":
	A = Matrix([
		[1, 2, 3],
		[0, 4, -1],
	])

	assert A.size == (2, 3)
	assert A.is_square == False
	assert A.is_line == False
	assert A.is_column == False

	B = Matrix([
		[1, 0, 0],
		[0, 2, 0],
		[0, 0, 0],
	])

	assert B.size == (3, 3)
	assert B.is_square == True
	assert B.is_line == False
	assert B.is_column == False

	C = Matrix([
		[1],
		[0],
		[-1],
	])

	assert C.size == (3, 1)
	assert C.is_square == False
	assert C.is_line == False
	assert C.is_column == True

	D = Matrix([
		[1, 1, -5]
	])

	assert D.size == (1, 3)
	assert D.is_square == False
	assert D.is_line == True
	assert D.is_column == False

	E = Matrix([
		[5, 5, 10],
		[0, 5, 5],
		[5, 5, 10],
	])

	F = Matrix([
		[5, 5, 0],
		[10, 5, 5],
		[5, 5, 0],
	])

	G = Matrix([
		[10, 10, 10],
		[10, 10, 10],
		[10, 10, 10],
	])

	GN = Matrix([
		[-10, -10, -10],
		[-10, -10, -10],
		[-10, -10, -10],
	])

	assert E != F
	assert E + F == G

	assert -G == GN

	print(A * B)

	assert A * B == Matrix([
		[1, 4, 0],
		[0, 8, 0]
	])

	assert A != B and B != C and C != B

	I2 = Matrix.identity(2)

	assert I2.size == (2, 2)
	assert I2.is_square == True
	assert I2.order == 2
	for ti in range(2):
		assert I2.diagonal[ti] == 1

	O2 = Matrix.null(2)

	assert O2.size == (2, 2)
	assert O2.is_square == True
	assert O2.order == 2
	for tidx in O2.iter():
		assert O2.index(tidx.i, tidx.j) == 0

# print(A[1, 1])
# print(A.size)
# print(A.order)
# print(A.diagonal)
# print(Matrix.identity(2))

# print(Matrix.null(2))

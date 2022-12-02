from pyctlib.basictype import vector
a = vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
a.map(lambda x: x * 2)
a.filter(lambda x: x % 2 == 0)
a.all(lambda x: x > 1)
a.any(lambda x: x == 2)
a[a > 7] = 7
a
a.reduce(lambda x, y: x + y)
a.index(lambda x: x % 4 == 0)

import unittest
import sys
import os
from zytlib.container.vector import vector, IndexMapping

class TestVector(unittest.TestCase):

    def test_init(self):
        a = vector([1, 2, 3])
        self.assertEqual(a.length, 3)
        self.assertEqual(list(a), [1, 2, 3])
        a = vector(1, 2, 3)
        self.assertEqual(a.length, 3)
        self.assertEqual(list(a), [1, 2, 3])
        a = vector((1, 2, 3))
        self.assertEqual(a.length, 3)
        self.assertEqual(list(a), [1, 2, 3])
        a = vector()
        self.assertEqual(a.length, 0)
        self.assertEqual(list(a), [])

    def test_sum(self):
        a = vector()
        self.assertEqual(a.sum(), None)
        self.assertEqual(a.sum(default=0), 0)
        a = vector(1, 2, 3)
        self.assertEqual(a.sum(), 6)

        vec = vector()
        for index in range(10):
            vec.append(index)
            self.assertEqual(vec.sum(), sum(_ for _ in range(index+1)))
            self.assertEqual(vec.element_type, int)
            self.assertEqual(vec.set(), set([_ for _ in range(index + 1)]))
            self.assertTrue(vec.ishashable())

        vec.append("hello")
        self.assertIsNone(vec.sum())
        self.assertEqual(vec.element_type, set([int, str]))
        self.assertEqual(vec.set(), set([_ for _ in range(10)] + ["hello"]))
        self.assertTrue(vec.ishashable())

    def test_sort(self):
        vec = vector.range(10).shuffle()
        vec.sort_()
        self.assertEqual(list(vec), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(vec.max(), 9)
        self.assertEqual(vec.min(), 0)
        self.assertEqual(list(vector([1, 2, 3, 4, 1]).sort_by_index(key=lambda x: -x)), [1, 4, 3, 2, 1])

    def test_map(self):
        self.assertEqual(list(vector([0, 1, 2]).map(lambda x: x ** 2)), [0, 1, 4])
        self.assertEqual(list(vector([[0, 1], [2, 3]], recursive=True).rmap(lambda x: x + 1).flatten()), [1, 2, 3, 4])

    def test_filter(self):
        self.assertEqual(list(vector([1, 2, 3, 4, 5, 6]).filter(lambda x: x > 3)), [4, 5, 6])

    def test_funcs(self):
        self.assertEqual(list(vector(0, 1, 2, 3).test(lambda x: 1 / x)), [1, 2, 3])
        self.assertEqual(list(vector(0, 1, 2, 3).testnot(lambda x: 1 / x)), [0])
        self.assertEqual(list(vector(0, 1, 2, 3, 1).replace(1, -1)), [0, -1, 2, 3, -1])
        self.assertEqual(list(vector(0, 1, 2, 3, 4).replace(lambda x: x > 2, 2)), [0, 1, 2, 2, 2])
        self.assertEqual(list(vector(0, 1, 2, 3, 4).replace(lambda x: x > 2, lambda x: x + 2)), [0, 1, 2, 5, 6])
        self.assertEqual(list(vector([1, 2, 3, 2, 3, 1]).unique()), [1, 2, 3])
        self.assertEqual(list(vector(0, 1, 2, 3, 4, 0, 1).findall_crash(lambda x: 1 / x)), [0, 5])
        self.assertEqual(list(vector([1, 2, 3, 4, 2, 3]).findall(lambda x: x > 2)), [2, 3, 5])

    def test_reshape(self):
        self.assertEqual(list(vector([1, 2, 3]) * vector([4, 5, 6])), [(1, 4), (2, 5), (3, 6)])
        self.assertEqual(list(vector([1, 2, 3]) ** vector([2, 3, 4])), [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)])
        self.assertEqual(list(vector.zip(vector(1, 2, 3), vector(1, 2, 3), vector(1, 2, 3))), [(1, 1, 1), (2, 2, 2), (3, 3, 3)])

    def test_shape(self):
        self.assertEqual(vector.range(100).reshape(2, -1).shape, (2, 50))

    def test_index_mapping(self):
        x = vector.range(100).shuffle()
        self.assertEqual(x, vector.range(100).map_index_from(x))
        x = vector.range(100).sample(10, replace=False)
        self.assertEqual(x, vector.range(100).map_index_from(x))
        x = vector.range(100)[:10]
        y = vector.range(100).map_index_from(x)
        self.assertEqual(x, y)
        self.assertEqual(list(vector.range(100)[::-1]), list(range(100)[::-1]))
        x = vector.rand(10)
        self.assertTrue(all(x[::-2][::-1] == x[1::2]))
        t1 = IndexMapping(slice(0, 15, 2), 10, True)
        t2 = IndexMapping([4,3,2,1,0], 5, True)
        self.assertEqual(list(t1.map(t2).index_map), [4, -1, 3, -1, 2, -1, 1, -1, 0, -1])
        t3 = IndexMapping(slice(0, 2, 1), range_size=5, reverse=True)
        self.assertEqual(list(t1.map(t3).index_map), [0, -1, 1, -1, -1, -1, -1, -1, -1, -1])

import unittest
from zytlib.container.table import table

class TestTable(unittest.TestCase):

    def test_init(self):
        a = table(x=1, y=2)
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertRaises(Exception, a.__getitem__, "z")
        self.assertRaises(Exception, a.__getitem__, "1")

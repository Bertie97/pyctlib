import unittest
from zytlib.container.table import table
from zytlib.container.torch_table import torch_table
import torch

class TestTable(unittest.TestCase):

    def test_init(self):
        a = table(x=1, y=2)
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertRaises(Exception, a.__getitem__, "z")
        self.assertRaises(Exception, a.__getitem__, "1")

    def test_torch_table(self):
        a = torch_table()
        a.concate("x", torch.zeros(10))
        self.assertEqual(a.x.dim(), 2)

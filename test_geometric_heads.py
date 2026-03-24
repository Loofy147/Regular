import unittest
import os
from decimal import Decimal
from geometric_heads import HeadTargeter

class TestGeometricHeads(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.targeter = HeadTargeter(dim=self.dim)

    def test_frequency_length(self):
        self.assertEqual(len(self.targeter.frequencies), self.dim // 2)

    def test_geometric_integrity(self):
        is_geo, ratio = self.targeter.verify_geometric_integrity()
        self.assertTrue(is_geo)

    def test_get_head_parameters(self):
        params = self.targeter.get_head_parameters(0, 5)
        self.assertEqual(params["head_index"], 0)
        self.assertEqual(params["position"], 5)
        self.assertEqual(params["frequency"], "1")

    def test_save_map(self):
        filename = "test_map.json"
        self.targeter.save_map(seq_len=2, filename=filename)
        self.assertTrue(os.path.exists(filename))
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    unittest.main()

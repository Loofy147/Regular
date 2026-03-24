import unittest
from decimal import Decimal
from geometric_heads import calculate_geometric_heads, verify_geometric_sequence

class TestGeometricHeads(unittest.TestCase):
    def test_calculate_geometric_heads(self):
        dim = 64
        base = 10000
        heads_map = calculate_geometric_heads(dim=dim, base=base)

        self.assertEqual(len(heads_map), dim // 2)
        self.assertEqual(heads_map[0], "1")

    def test_verify_geometric_sequence(self):
        # Valid geometric sequence
        valid_map = {0: "1", 1: "0.5", 2: "0.25"}
        is_geo, ratio = verify_geometric_sequence(valid_map)
        self.assertTrue(is_geo)
        self.assertEqual(ratio, Decimal("0.5"))

        # Invalid geometric sequence
        invalid_map = {0: "1", 1: "0.5", 2: "0.1"}
        is_geo, ratio = verify_geometric_sequence(invalid_map)
        self.assertFalse(is_geo)

if __name__ == "__main__":
    unittest.main()

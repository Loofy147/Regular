import unittest
from decimal import Decimal
from genieune_heads.core import GeometricOrbitConstructor, GeometricHeadTargeter

class TestGeometricConstruction(unittest.TestCase):
    def test_spike_function(self):
        # m=7, v=0.1, j0=2, delta=0.5
        # Spike at j=2 should be 0.1 + 0.5 = 0.6
        # Other j should be 0.1
        constructor = GeometricOrbitConstructor(m=7)

        val_spike = constructor.spike_function(2, 0.1, 2, 0.5)
        self.assertAlmostEqual(float(val_spike), 0.6)

        val_other = constructor.spike_function(0, 0.1, 2, 0.5)
        self.assertAlmostEqual(float(val_other), 0.1)

    def test_genuine_head_computation(self):
        # r=1, v=0, j0=0, delta=1, m=3
        # orbit: [1, 0, 0] (j=0 spike, others 0)
        # sigma = 1
        # head_start = (r * sigma) % m = (1 * 1) % 3 = 1
        constructor = GeometricOrbitConstructor(m=3)
        params = [{"r": 1, "v": 0, "j0": 0, "delta": 1}]
        results = constructor.compute_genuine_heads(params)

        self.assertEqual(results[0]["sigma"], "1")
        self.assertEqual(results[0]["head_start"], "1")
        self.assertEqual(results[0]["orbit"], ["1", "0", "0"])

    def test_geometric_targeter(self):
        targeter = GeometricHeadTargeter(m=5)
        params = [
            {"r": 1, "v": 0.1, "j0": 0, "delta": 0.5},
            {"r": 2, "v": 0.2, "j0": 1, "delta": 0.5},
            {"r": 3, "v": 0.3, "j0": 2, "delta": 0.5}
        ]
        targeter.set_orbit_parameters(params)

        head0 = targeter.get_genuine_head(0)
        self.assertIsNotNone(head0)
        self.assertEqual(head0["color"], 0)

        head2 = targeter.get_genuine_head(2)
        self.assertEqual(head2["color"], 2)

if __name__ == "__main__":
    unittest.main()

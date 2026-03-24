import unittest
from decimal import Decimal
from geometric_heads import HeadTargeter, PressureWeightSystem, Modulator

class TestGeometricHeads(unittest.TestCase):
    def test_pressure_weights(self):
        n_heads = 4
        # base_pressure = 8, so ratio should be 2^(-8/4) = 2^-2 = 0.25
        system = PressureWeightSystem(n_heads=n_heads)
        is_geo, ratio = system.verify_geometric_ratio()
        self.assertTrue(is_geo)
        self.assertAlmostEqual(float(ratio), 0.25, places=15)

    def test_modulator(self):
        targeter = HeadTargeter(dim=32)
        system = PressureWeightSystem(n_heads=8)
        modulator = Modulator(targeter, system)

        params = modulator.target_head(0, 5)
        self.assertIn("pressure_weight", params)
        self.assertEqual(params["frequency"], "1")

    def test_pressure_weight_range(self):
        system = PressureWeightSystem(n_heads=8)
        self.assertEqual(len(system.weights), 8)
        with self.assertRaises(ValueError):
            system.get_weight(10)

if __name__ == "__main__":
    unittest.main()

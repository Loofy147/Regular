import unittest
import os
import json
from decimal import Decimal
from genieune_heads import (
    HeadTargeter,
    PressureWeightSystem,
    Modulator,
    HeadsMapCache,
    HeadAnalyzer,
    SequenceEncoder,
    AttentionBiasMatrix,
    GenieuneHeadsProfiler
)

class TestUpgrades(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.n_heads = 8
        self.targeter = HeadTargeter(dim=self.dim)
        self.pressure_system = PressureWeightSystem(n_heads=self.n_heads)
        self.modulator = Modulator(self.targeter, self.pressure_system)

    def test_heads_map_cache(self):
        cache_file = "test_heads_map.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)

        cache = HeadsMapCache(cache_file)
        params = {"sin": "0.5", "cos": "0.866", "head_index": 0, "position": 1}
        cache.update(0, 1, params)
        cache.save()

        self.assertTrue(os.path.exists(cache_file))

        new_cache = HeadsMapCache(cache_file)
        cached_params = new_cache.get(0, 1)
        self.assertEqual(cached_params["sin"], "0.5")
        self.assertEqual(cached_params["decimal_sin"], Decimal("0.5"))

        os.remove(cache_file)

    def test_head_analyzer(self):
        analyzer = HeadAnalyzer(self.targeter)
        bands = analyzer.analyze_frequency_bands(n_bands=2)
        self.assertEqual(len(bands["bands"]), 2)

        trajectory = analyzer.generate_phase_portrait(0, max_pos=10)
        self.assertEqual(len(trajectory), 10)

        entropy = analyzer.calculate_band_entropy()
        self.assertGreaterEqual(entropy, 0)

    def test_sequence_encoder(self):
        encoder = SequenceEncoder(self.targeter)
        seq_len = 5
        matrix = encoder.build_embedding_matrix(seq_len)
        self.assertEqual(len(matrix), seq_len)
        self.assertEqual(len(matrix[0]), self.dim)

    def test_attention_bias_matrix(self):
        bias_gen = AttentionBiasMatrix(self.pressure_system)
        seq_len = 4
        bias = bias_gen.build_full_bias_matrix(seq_len)
        self.assertEqual(len(bias), self.n_heads)
        self.assertEqual(len(bias[0]), seq_len)
        self.assertEqual(len(bias[0][0]), seq_len)

        # Check ALiBi distance property
        # Head 0, pos (0,1) should be -slope * |0-1|
        slope = self.pressure_system.get_weight(0)
        self.assertEqual(bias[0][0][1], -slope)

    def test_modulator_enhancements(self):
        seq_len = 3
        matrix = self.modulator.encode_sequence(seq_len)
        self.assertEqual(len(matrix), seq_len)

        bias = self.modulator.build_attention_bias(seq_len)
        self.assertEqual(len(bias), self.n_heads)

        batch = [[Decimal('1.0')] * self.dim, [Decimal('0.5')] * self.dim]
        rotated_batch = self.modulator.apply_rope_batch(batch, 5)
        self.assertEqual(len(rotated_batch), 2)
        self.assertEqual(len(rotated_batch[0]), self.dim)

    def test_profiler(self):
        profiler = GenieuneHeadsProfiler()
        profiler.profile_call(self.targeter.get_head_parameters, 0, 1)
        report = profiler.get_report()
        self.assertIn("get_head_parameters", report)

if __name__ == "__main__":
    unittest.main()

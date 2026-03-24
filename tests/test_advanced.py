import unittest
from decimal import Decimal
from genieune_heads import (
    HeadTargeter,
    PressureWeightSystem,
    Modulator,
    StreamingEncoder,
    HeadAnalyzer
)

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.n_heads = 8
        self.targeter = HeadTargeter(dim=self.dim)
        self.pressure_system = PressureWeightSystem(n_heads=self.n_heads)
        self.modulator = Modulator(self.targeter, self.pressure_system)

    def test_apply_rope_sequence(self):
        seq = [[Decimal('1.0')] * self.dim for _ in range(5)]
        rotated_seq = self.modulator.apply_rope_sequence(seq)
        self.assertEqual(len(rotated_seq), 5)

        # Check first vector at position 0 (should not be rotated if freq 0, pos 0)
        # But frequencies are (base)**(-2i/d), freq[0] is (10000)**(0) = 1.0
        # Wait, for pos=0, sin(0)=0, cos(0)=1, so x1*1 - x2*0 = x1, x1*0 + x2*1 = x2
        # Position 0 should always be identity for RoPE
        self.assertEqual(rotated_seq[0][0], Decimal('1.0'))
        self.assertEqual(rotated_seq[0][1], Decimal('1.0'))

        # Position 1 should be rotated
        self.assertNotEqual(rotated_seq[1][0], Decimal('1.0'))

    def test_streaming_encoder(self):
        streamer = StreamingEncoder(self.targeter)
        vec = [Decimal('1.0')] * self.dim

        # Pos 0
        rot0 = streamer.encode_next(vec)
        self.assertEqual(rot0[0], Decimal('1.0'))
        self.assertEqual(streamer.current_pos, 1)

        # Pos 1
        rot1 = streamer.encode_next(vec)
        self.assertNotEqual(rot1[0], Decimal('1.0'))
        self.assertEqual(streamer.current_pos, 2)

        # Reset
        streamer.reset(5)
        self.assertEqual(streamer.current_pos, 5)

    def test_modulate_attention(self):
        scores = [[Decimal('1.0'), Decimal('0.5')], [Decimal('0.5'), Decimal('1.0')]]
        modulated = self.modulator.modulate_attention(scores, 0)
        self.assertEqual(len(modulated), 2)

        # ALiBi style penalty: score - slope * distance
        # head 0, pos (0,1) dist is 1, so modulated[0][1] = 0.5 - slope
        slope = self.pressure_system.get_weight(0)
        self.assertEqual(modulated[0][1], Decimal('0.5') - slope)

    def test_head_analyzer_expanded(self):
        analyzer = HeadAnalyzer(self.targeter)
        drift = analyzer.calculate_phase_drift(0, seq_len=10)
        self.assertLess(drift, Decimal('1e-90')) # Precision targeting stability

        harmonics = analyzer.identify_harmonic_heads()
        # Should be some harmonics in geometric sequence with base 10000
        # theta_i = 10000^(-2i/d). ratio theta_i / theta_j = 10000^(-2(i-j)/d)
        # For ratio to be integer, (i-j) must be such that 10000^(2(j-i)/d) is integer
        self.assertIsInstance(harmonics, list)

if __name__ == "__main__":
    unittest.main()

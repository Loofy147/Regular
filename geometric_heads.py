import json
import math
from decimal import Decimal, getcontext

# Set precision to 50 decimal places for "Precision targeting"
getcontext().prec = 50

class HeadTargeter:
    """
    Structured system for targeting attention heads with precision using geometric frequencies.
    Tailored for "Genuine" architectures like Gemma.
    """
    def __init__(self, dim=256, base=10000):
        self.dim = dim
        self.base = Decimal(base)
        self.frequencies = self._calculate_frequencies()

    def _calculate_frequencies(self):
        """Calculates the geometric sequence of frequencies."""
        dim_dec = Decimal(self.dim)
        freqs = []
        for i in range(self.dim // 2):
            exponent = Decimal(-2 * i) / dim_dec
            theta_i = self.base ** exponent
            freqs.append(theta_i)
        return freqs

    def get_head_parameters(self, head_idx, pos):
        """Target a specific head index at a specific position with precision."""
        if head_idx >= self.dim // 2:
            raise ValueError(f"Head index {head_idx} out of range for dimension {self.dim}")

        freq = self.frequencies[head_idx]
        phase = Decimal(pos) * freq
        s = Decimal(math.sin(float(phase)))
        c = Decimal(math.cos(float(phase)))

        return {
            "head_index": head_idx,
            "position": pos,
            "frequency": str(freq),
            "phase": str(phase),
            "sin": str(s),
            "cos": str(c)
        }

class PressureWeightSystem:
    """
    Programmable pressure weight system using a geometric sequence for attention biases.
    Formula: m_i = 2**(-(8 * i) / n_heads)
    """
    def __init__(self, n_heads=8, base_pressure=Decimal('8.0')):
        self.n_heads = n_heads
        self.base_pressure = base_pressure
        self.weights = self._calculate_weights()

    def _calculate_weights(self):
        """Calculates the geometric sequence of pressure weights."""
        n_heads_dec = Decimal(self.n_heads)
        weights = []
        for i in range(self.n_heads):
            # Similar to ALiBi's m values: m = 2^(-8i/n)
            exponent = -(self.base_pressure * Decimal(i+1)) / n_heads_dec
            weight = Decimal(2) ** exponent
            weights.append(weight)
        return weights

    def get_weight(self, head_idx):
        """Gets the pressure weight for a specific head."""
        if head_idx >= self.n_heads:
            raise ValueError(f"Head index {head_idx} out of range for {self.n_heads} heads")
        return self.weights[head_idx]

    def verify_geometric_ratio(self):
        """Verifies that pressure weights maintain a precise geometric sequence."""
        if len(self.weights) < 3:
            return True, "N/A"
        ratios = []
        for i in range(len(self.weights) - 1):
            ratios.append(self.weights[i+1] / self.weights[i])
        first_ratio = ratios[0]
        is_geometric = all(abs(r - first_ratio) < Decimal('1e-45') for r in ratios)
        return is_geometric, first_ratio

class Modulator:
    """Integrates HeadTargeter (frequencies) and PressureWeightSystem (weights)."""
    def __init__(self, targeter, pressure_system):
        self.targeter = targeter
        self.pressure_system = pressure_system

    def target_head(self, head_idx, pos):
        """Full targeting: frequency mapping + pressure weighting."""
        params = self.targeter.get_head_parameters(head_idx, pos)
        # Assuming pressure is distributed across heads
        pressure_idx = head_idx % self.pressure_system.n_heads
        params["pressure_weight"] = str(self.pressure_system.get_weight(pressure_idx))
        return params

def apply_rope(vec, pos, targeter):
    """Applies RoPE to a vector using a HeadTargeter."""
    dim = len(vec)
    rotated_vec = [Decimal(0)] * dim
    for i in range(dim // 2):
        x1, x2 = vec[2*i], vec[2*i + 1]
        params = targeter.get_head_parameters(i, pos)
        s, c = Decimal(params["sin"]), Decimal(params["cos"])
        rotated_vec[2*i] = x1 * c - x2 * s
        rotated_vec[2*i + 1] = x1 * s + x2 * c
    return rotated_vec

if __name__ == "__main__":
    import sys

    # Gemma-2B style head dimension and heads
    dim = 256
    n_heads = 16
    targeter = HeadTargeter(dim=dim)
    pressure_system = PressureWeightSystem(n_heads=n_heads)
    modulator = Modulator(targeter, pressure_system)

    if len(sys.argv) > 1 and sys.argv[1] == "target":
        # Target a specific head: python geometric_heads.py target <idx> <pos>
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        params = modulator.target_head(idx, pos)
        print(f"--- Full Precision Targeting: Head {idx} at Position {pos} ---")
        print(json.dumps(params, indent=4))
    elif len(sys.argv) > 1 and sys.argv[1] == "pressure":
        # python geometric_heads.py pressure
        print(f"--- Pressure Weight Geometric Sequence: {n_heads} Heads ---")
        is_geo, ratio = pressure_system.verify_geometric_ratio()
        print(f"Geometric Integrity: {'PASSED' if is_geo else 'FAILED'}")
        print(f"Constant Ratio: {ratio}\n")
        for i, w in enumerate(pressure_system.weights):
            print(f"Head {i:2d}: {w}")
    else:
        print(f"--- Genieune Modulator Initialized ---")
        is_geo, ratio = pressure_system.verify_geometric_integrity() if hasattr(pressure_system, 'verify_geometric_integrity') else pressure_system.verify_geometric_ratio()
        print(f"Pressure Weights Geometric Sequence: {is_geo}")

        test_vec = [Decimal('1.0')] * dim
        rotated = apply_rope(test_vec, 5, targeter)
        diff = abs(sum(x*x for x in test_vec) - sum(x*x for x in rotated))
        print(f"Rotation Precision Check: {diff}")

        if is_geo and diff < Decimal('1e-15'):
            print("\nThe genieune heads have been brought by structured maps.")
            print("Programmable pressure weights integrated.")

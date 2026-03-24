import json
import math
from decimal import Decimal, getcontext

# Set precision very high for "Precision targeting"
getcontext().prec = 100

def decimal_sin_cos(x, terms=50):
    """
    Calculates sin(x) and cos(x) using Taylor series for high precision.
    x is a Decimal.
    """
    # Normalize x to [-pi, pi] for better convergence
    pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
    two_pi = 2 * pi
    x = x % two_pi
    if x > pi:
        x -= two_pi

    sin_x = Decimal(0)
    cos_x = Decimal(0)

    term = x
    x_sq = x * x

    # Sin series: x - x^3/3! + x^5/5! ...
    for i in range(terms):
        sin_x += term
        term *= -x_sq / Decimal((2*i + 2) * (2*i + 3))

    # Cos series: 1 - x^2/2! + x^4/4! ...
    term = Decimal(1)
    for i in range(terms):
        cos_x += term
        term *= -x_sq / Decimal((2*i + 1) * (2*i + 2))

    return sin_x, cos_x

class HeadTargeter:
    """
    Structured system for targeting attention heads with precision using geometric frequencies.
    Tailored for "Genuine" architectures like Gemma.
    """
    def __init__(self, dim=256, base=10000):
        self.dim = dim
        self.base = Decimal(base)
        self._frequencies = None
        self._param_cache = {}

    @property
    def frequencies(self):
        """Calculates and caches the geometric sequence of frequencies."""
        if self._frequencies is None:
            dim_dec = Decimal(self.dim)
            self._frequencies = []
            for i in range(self.dim // 2):
                exponent = Decimal(-2 * i) / dim_dec
                theta_i = self.base ** exponent
                self._frequencies.append(theta_i)
        return self._frequencies

    def get_head_parameters(self, head_idx, pos):
        """Target a specific head index at a specific position with precision (cached)."""
        cache_key = (head_idx, pos)
        if cache_key in self._param_cache:
            return self._param_cache[cache_key]

        if head_idx >= self.dim // 2:
            raise ValueError(f"Head index {head_idx} out of range for dimension {self.dim}")

        freq = self.frequencies[head_idx]
        phase = Decimal(pos) * freq

        s, c = decimal_sin_cos(phase)

        params = {
            "head_index": head_idx,
            "position": pos,
            "frequency": str(freq),
            "phase": str(phase),
            "sin": str(s),
            "cos": str(c),
            "decimal_sin": s,
            "decimal_cos": c
        }
        self._param_cache[cache_key] = params
        return params

    def verify_geometric_integrity(self):
        """Verifies that frequencies follow a precise geometric sequence."""
        freqs = self.frequencies
        if len(freqs) < 3:
            return True, "N/A"
        ratios = [freqs[i+1] / freqs[i] for i in range(len(freqs)-1)]
        first_ratio = ratios[0]
        is_geometric = all(abs(r - first_ratio) < Decimal('1e-90') for r in ratios)
        return is_geometric, first_ratio

class PressureWeightSystem:
    """
    Programmable pressure weight system using a geometric sequence for attention biases.
    """
    def __init__(self, n_heads=8, base_pressure=Decimal('8.0')):
        self.n_heads = n_heads
        self.base_pressure = base_pressure
        self._weights = None

    @property
    def weights(self):
        if self._weights is None:
            n_heads_dec = Decimal(self.n_heads)
            self._weights = [Decimal(2) ** (-(self.base_pressure * Decimal(i+1)) / n_heads_dec) for i in range(self.n_heads)]
        return self._weights

    def get_weight(self, head_idx):
        """Gets the pressure weight for a specific head."""
        if head_idx < 0 or head_idx >= self.n_heads:
            raise ValueError(f"Head index {head_idx} out of range for {self.n_heads} heads")
        return self.weights[head_idx]

    def verify_geometric_ratio(self):
        w = self.weights
        if len(w) < 3:
            return True, "N/A"
        ratios = [w[i+1] / w[i] for i in range(len(w)-1)]
        first_ratio = ratios[0]
        is_geometric = all(abs(r - first_ratio) < Decimal('1e-90') for r in ratios)
        return is_geometric, first_ratio

class Modulator:
    """Integrates HeadTargeter (frequencies) and PressureWeightSystem (weights)."""
    def __init__(self, targeter, pressure_system):
        self.targeter = targeter
        self.pressure_system = pressure_system

    def target_head(self, head_idx, pos):
        """Full targeting: frequency mapping + pressure weighting."""
        params = self.targeter.get_head_parameters(head_idx, pos)
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
        s, c = params["decimal_sin"], params["decimal_cos"]
        rotated_vec[2*i] = x1 * c - x2 * s
        rotated_vec[2*i + 1] = x1 * s + x2 * c
    return rotated_vec

if __name__ == "__main__":
    import sys
    dim = 256
    targeter = HeadTargeter(dim=dim)
    pressure_system = PressureWeightSystem(n_heads=16)
    modulator = Modulator(targeter, pressure_system)

    if len(sys.argv) > 1 and sys.argv[1] == "target":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        params = modulator.target_head(idx, pos)
        print(json.dumps({k:v for k,v in params.items() if not k.startswith("decimal_")}, indent=4))
    elif len(sys.argv) > 1 and sys.argv[1] == "pressure":
        print(f"--- Pressure Weight Sequence ---")
        for i, w in enumerate(pressure_system.weights):
            print(f"Head {i:2d}: {w}")
    else:
        print(f"--- Genieune Modulator Initialized (Ultra High Precision) ---")
        is_geo, _ = targeter.verify_geometric_integrity()
        print(f"Frequency Geometric Integrity: {is_geo}")
        test_vec = [Decimal('1.0')] * dim
        rotated = apply_rope(test_vec, 5, targeter)
        diff = abs(sum(x*x for x in test_vec) - sum(x*x for x in rotated))
        print(f"Precision Check (Norm Diff): {diff}")

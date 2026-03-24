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
        """
        Target a specific head index at a specific position with precision.
        Returns frequency, phase, sin, and cos values.
        """
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

    def generate_structured_map(self, seq_len):
        """Generates a complete structured map for a sequence of positions."""
        structured_map = {}
        for pos in range(seq_len):
            pos_map = {}
            for i in range(self.dim // 2):
                pos_map[i] = self.get_head_parameters(i, pos)
            structured_map[pos] = pos_map
        return structured_map

    def find_target(self, target_value, precision=Decimal('0.01'), max_pos=100):
        """
        Finds the first head/position pair that targets a specific sin value with precision.
        """
        target = Decimal(target_value)
        for pos in range(max_pos):
            for i in range(self.dim // 2):
                params = self.get_head_parameters(i, pos)
                if abs(Decimal(params["sin"]) - target) < precision:
                    return params
        return None

    def visualize_frequencies(self, width=60):
        """Provides an ASCII visualization of the geometric sequence."""
        print("\n--- Geometric Frequency Sequence Visualization ---")
        max_f = float(self.frequencies[0])
        for i, f in enumerate(self.frequencies[:16]): # Show first 16
            bar_len = int((float(f) / max_f) * width)
            print(f"Head {i:2d}: [{'#' * bar_len}{' ' * (width - bar_len)}] {float(f):.6f}")
        print("...")

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

    # Gemma-2B style head dimension
    dim = 256
    targeter = HeadTargeter(dim=dim)

    if len(sys.argv) > 1 and sys.argv[1] == "target":
        if len(sys.argv) > 2 and sys.argv[2] == "value":
            # Search for a specific value: python geometric_heads.py target value 0.5
            val = sys.argv[3] if len(sys.argv) > 3 else "0.5"
            print(f"--- Precision Targeting for Value: {val} ---")
            found = targeter.find_target(val)
            if found:
                print(json.dumps(found, indent=4))
            else:
                print("No match found within search bounds.")
        else:
            # Target a specific head: python geometric_heads.py target <idx> <pos>
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            params = targeter.get_head_parameters(idx, pos)
            print(f"--- Precision Targeting: Head {idx} at Position {pos} ---")
            print(json.dumps(params, indent=4))
    else:
        print(f"--- Genieune Head System: {dim} Dimensions ---")
        targeter.visualize_frequencies()

        test_vec = [Decimal('1.0')] * dim
        rotated = apply_rope(test_vec, 5, targeter)
        diff = abs(sum(x*x for x in test_vec) - sum(x*x for x in rotated))
        print(f"\nPrecision Integrity Check (Norm Diff): {diff}")

        if diff < Decimal('1e-15'):
            print("\nThe genieune heads have been brought by structured maps in a geometric sequence.")
            print("Targeting system ready.")

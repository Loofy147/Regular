import json
import math
import time
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

class HeadsMapCache:
    """Loads and saves precomputed head parameters from/to JSON."""
    def __init__(self, filepath="heads_map.json"):
        self.filepath = filepath
        self.data = {}
        self.load()

    def load(self):
        try:
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def save(self, filepath=None):
        target = filepath or self.filepath
        with open(target, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get(self, head_idx, pos):
        pos_str = str(pos)
        head_idx_str = str(head_idx)
        if pos_str in self.data and head_idx_str in self.data[pos_str]:
            params = self.data[pos_str][head_idx_str].copy()
            # Convert sin/cos back to Decimal for use in apply_rope
            params["decimal_sin"] = Decimal(params["sin"])
            params["decimal_cos"] = Decimal(params["cos"])
            return params
        return None

    def update(self, head_idx, pos, params):
        pos_str = str(pos)
        head_idx_str = str(head_idx)
        if pos_str not in self.data:
            self.data[pos_str] = {}
        # Store only what's needed for JSON, avoid Decimal objects
        json_params = {k: v for k, v in params.items() if not k.startswith("decimal_")}
        self.data[pos_str][head_idx_str] = json_params

class HeadTargeter:
    """
    Structured system for targeting attention heads with precision using geometric frequencies.
    Tailored for "Genuine" architectures like Gemma.
    """
    def __init__(self, dim=256, base=10000, cache_file=None):
        self.dim = dim
        self.base = Decimal(base)
        self._frequencies = None
        self._param_cache = {}
        self.external_cache = HeadsMapCache(cache_file) if cache_file else None

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

        if self.external_cache:
            cached = self.external_cache.get(head_idx, pos)
            if cached:
                self._param_cache[cache_key] = cached
                return cached

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

        if self.external_cache:
            self.external_cache.update(head_idx, pos, params)

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

class SequenceEncoder:
    """Encodes full sequences of positions into embedding matrices."""
    def __init__(self, targeter):
        self.targeter = targeter

    def encode_sequence(self, seq_len):
        """Encodes a full sequence of positions (seq_len, dim/2)."""
        matrix = []
        for pos in range(seq_len):
            row = []
            for head_idx in range(self.targeter.dim // 2):
                params = self.targeter.get_head_parameters(head_idx, pos)
                row.append((params["decimal_sin"], params["decimal_cos"]))
            matrix.append(row)
        return matrix

    def build_embedding_matrix(self, seq_len):
        """Builds a full embedding matrix (seq_len, dim)."""
        embedding = []
        for pos in range(seq_len):
            row = [Decimal(0)] * self.targeter.dim
            for head_idx in range(self.targeter.dim // 2):
                params = self.targeter.get_head_parameters(head_idx, pos)
                row[2*head_idx] = params["decimal_sin"]
                row[2*head_idx + 1] = params["decimal_cos"]
            embedding.append(row)
        return embedding

class AttentionBiasMatrix:
    """Generates ALiBi-style bias matrices using pressure weights."""
    def __init__(self, pressure_system):
        self.pressure_system = pressure_system

    def build_full_bias_matrix(self, seq_len):
        """Builds a full [n_heads, seq_len, seq_len] bias matrix."""
        n_heads = self.pressure_system.n_heads
        bias_matrix = []
        for h in range(n_heads):
            slope = self.pressure_system.get_weight(h)
            head_matrix = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    # ALiBi style distance penalty: -slope * |i - j|
                    dist = Decimal(abs(i - j))
                    row.append(-(slope * dist))
                head_matrix.append(row)
            bias_matrix.append(head_matrix)
        return bias_matrix

    def get_bias_for_head(self, head_idx, seq_len):
        """Generates a bias matrix for a single head."""
        slope = self.pressure_system.get_weight(head_idx % self.pressure_system.n_heads)
        head_matrix = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                dist = Decimal(abs(i - j))
                row.append(-(slope * dist))
            head_matrix.append(row)
        return head_matrix

class Modulator:
    """Integrates HeadTargeter (frequencies) and PressureWeightSystem (weights)."""
    def __init__(self, targeter, pressure_system):
        self.targeter = targeter
        self.pressure_system = pressure_system
        self.encoder = SequenceEncoder(targeter)
        self.bias_generator = AttentionBiasMatrix(pressure_system)

    def target_head(self, head_idx, pos):
        """Full targeting: frequency mapping + pressure weighting."""
        params = self.targeter.get_head_parameters(head_idx, pos)
        pressure_idx = head_idx % self.pressure_system.n_heads
        params["pressure_weight"] = str(self.pressure_system.get_weight(pressure_idx))
        return params

    def encode_sequence(self, seq_len):
        """Encodes a full sequence of positions."""
        return self.encoder.build_embedding_matrix(seq_len)

    def build_attention_bias(self, seq_len):
        """Generates the full ALiBi-style bias matrix."""
        return self.bias_generator.build_full_bias_matrix(seq_len)

    def apply_rope_batch(self, batch_vecs, pos):
        """Applies RoPE to a batch of vectors at a specific position."""
        return [apply_rope(vec, pos, self.targeter) for vec in batch_vecs]

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

class HeadAnalyzer:
    """Provides analysis of head behavior patterns."""
    def __init__(self, targeter):
        self.targeter = targeter

    def analyze_frequency_bands(self, n_bands=4):
        """Categorizes heads into frequency bands."""
        freqs = [float(f) for f in self.targeter.frequencies]
        min_f, max_f = min(freqs), max(freqs)
        band_size = (max_f - min_f) / n_bands if n_bands > 0 else 0

        bands = [[] for _ in range(n_bands)]
        for i, f in enumerate(freqs):
            band_idx = min(int((f - min_f) / band_size) if band_size > 0 else 0, n_bands - 1)
            bands[band_idx].append(i)

        return {
            "min_freq": min_f,
            "max_freq": max_f,
            "bands": bands
        }

    def generate_phase_portrait(self, head_idx, max_pos=100):
        """Generates (sin, cos) trajectory for a head."""
        trajectory = []
        for pos in range(max_pos):
            params = self.targeter.get_head_parameters(head_idx, pos)
            trajectory.append((float(params["sin"]), float(params["cos"])))
        return trajectory

    def calculate_band_entropy(self, n_bands=10):
        """Calculates entropy of frequency distribution across bands."""
        freqs = [float(f) for f in self.targeter.frequencies]
        min_f, max_f = min(freqs), max(freqs)
        if max_f == min_f:
            return 0.0

        hist = [0] * n_bands
        for f in freqs:
            idx = min(int((f - min_f) / (max_f - min_f) * n_bands), n_bands - 1)
            hist[idx] += 1

        total = len(freqs)
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

class GenieuneHeadsProfiler:
    """Built-in profiling for measuring core component performance."""
    def __init__(self):
        self.stats = {}

    def profile_call(self, func, *args, **kwargs):
        """Measures execution time of a function."""
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        name = func.__qualname__
        if name not in self.stats:
            self.stats[name] = {"count": 0, "total_time": 0.0, "max_time": 0.0, "min_time": float('inf')}

        stat = self.stats[name]
        stat["count"] += 1
        stat["total_time"] += duration
        stat["max_time"] = max(stat["max_time"], duration)
        stat["min_time"] = min(stat["min_time"], duration)

        return result

    def get_report(self):
        """Generates a summary profiling report."""
        report = []
        for name, s in self.stats.items():
            avg = s["total_time"] / s["count"] if s["count"] > 0 else 0
            report.append(f"{name:30s} | Avg: {avg:.6f}s | Max: {s['max_time']:.6f}s | Min: {s['min_time']:.6f}s | Count: {s['count']}")
        return "\n".join(report)

if __name__ == "__main__":
    import sys
    dim = 256
    targeter = HeadTargeter(dim=dim, cache_file="heads_map.json")
    pressure_system = PressureWeightSystem(n_heads=16)
    modulator = Modulator(targeter, pressure_system)
    analyzer = HeadAnalyzer(targeter)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "target":
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            params = modulator.target_head(idx, pos)
            print(json.dumps({k:v for k,v in params.items() if not k.startswith("decimal_")}, indent=4))
        elif cmd == "pressure":
            print(f"--- Pressure Weight Sequence ---")
            for i, w in enumerate(pressure_system.weights):
                print(f"Head {i:2d}: {w}")
        elif cmd == "encode":
            seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            matrix = modulator.encode_sequence(seq_len)
            print(f"--- Sequence Embedding Matrix (SeqLen={seq_len}) ---")
            for pos, row in enumerate(matrix):
                print(f"Pos {pos:2d}: {row[0]}") # Show first head for brevity
        elif cmd == "bias":
            seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            bias = modulator.build_attention_bias(seq_len)
            print(f"--- Attention Bias Matrix (Head 0, SeqLen={seq_len}) ---")
            for row in bias[0]:
                print([float(x) for x in row])
        elif cmd == "analyze":
            print(f"--- Head Analysis ---")
            bands = analyzer.analyze_frequency_bands()
            print(f"Frequency Bands: {[len(b) for b in bands['bands']]}")
            entropy = analyzer.calculate_band_entropy()
            print(f"Band Entropy: {entropy:.4f}")
        elif cmd == "export-map":
            target_file = sys.argv[2] if len(sys.argv) > 2 else "heads_map_exported.json"
            # Force compute a few positions if cache is empty or just export current
            if targeter.external_cache:
                targeter.external_cache.save(target_file)
                print(f"Exported heads map to {target_file}")
            else:
                print("No external cache to export.")
    else:
        print(f"--- Genieune Modulator Initialized (Ultra High Precision) ---")
        is_geo, _ = targeter.verify_geometric_integrity()
        print(f"Frequency Geometric Integrity: {is_geo}")
        test_vec = [Decimal('1.0')] * dim
        rotated = apply_rope(test_vec, 5, targeter)
        diff = abs(sum(x*x for x in test_vec) - sum(x*x for x in rotated))
        print(f"Precision Check (Norm Diff): {diff}")

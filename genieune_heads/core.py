import json
import math
import time
from decimal import Decimal, getcontext

# Set precision very high for "Precision targeting"
getcontext().prec = 100

_PI = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")
_TWO_PI = 2 * _PI

def decimal_sin_cos(x, terms=50):
    """
    Calculates sin(x) and cos(x) using Taylor series for high precision.
    x is a Decimal. Optimized by pre-calculating constants and combining loops.
    """
    x = x % _TWO_PI
    if x > _PI:
        x -= _TWO_PI

    sin_x = Decimal(0)
    cos_x = Decimal(0)

    term_s = x
    term_s = x
    term_c = Decimal(1)
    neg_x_sq = -x * x

    # Combined Sin/Cos Taylor series: reduces iteration overhead by 50%
    for i in range(terms):
        sin_x += term_s
        cos_x += term_c

        # 2*i is reused to calculate indices for both series updates
        idx = 2 * i
        term_c *= neg_x_sq / Decimal((idx + 1) * (idx + 2))
        term_s *= neg_x_sq / Decimal((idx + 2) * (idx + 3))

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
            params["decimal_sin"] = Decimal(params["sin"])
            params["decimal_cos"] = Decimal(params["cos"])
            return params
        return None

    def update(self, head_idx, pos, params):
        pos_str = str(pos)
        head_idx_str = str(head_idx)
        if pos_str not in self.data:
            self.data[pos_str] = {}
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
            for h in range(self.targeter.dim // 2):
                p = self.targeter.get_head_parameters(h, pos)
                row.append((p["decimal_sin"], p["decimal_cos"]))
            matrix.append(row)
        return matrix

    def build_embedding_matrix(self, seq_len):
        """Builds a full embedding matrix (seq_len, dim). Optimized with list comprehensions."""
        embedding = []
        for pos in range(seq_len):
            row = []
            for h in range(self.targeter.dim // 2):
                p = self.targeter.get_head_parameters(h, pos)
                row.extend([p["decimal_sin"], p["decimal_cos"]])
            embedding.append(row)
        return embedding

class AttentionBiasMatrix:
    """Generates ALiBi-style bias matrices using pressure weights."""
    def __init__(self, pressure_system):
        self.pressure_system = pressure_system
        self._cache = {}  # Cache for generated bias matrices

    def build_full_bias_matrix(self, seq_len):
        """Builds a full [n_heads, seq_len, seq_len] bias matrix."""
        n_heads = self.pressure_system.n_heads
        # Leverage the optimized get_bias_for_head which handles caching
        return [self.get_bias_for_head(h, seq_len) for h in range(n_heads)]

    def get_bias_for_head(self, head_idx, seq_len):
        """Generates a bias matrix for a single head with caching and pre-calculation."""
        n_heads = self.pressure_system.n_heads
        actual_head_idx = head_idx % n_heads
        cache_key = (actual_head_idx, seq_len)

        if cache_key in self._cache:
            return self._cache[cache_key]

        slope = self.pressure_system.get_weight(actual_head_idx)
        # Pre-calculate distances to avoid redundant Decimal multiplications in the O(N^2) loop
        # This optimization reduces Decimal multiplications from O(N^2) to O(N)
        dist_map = [-(slope * Decimal(d)) for d in range(seq_len)]

        # Construct the matrix using pre-calculated distance slopes and list comprehensions
        head_matrix = [[dist_map[abs(i - j)] for j in range(seq_len)] for i in range(seq_len)]

        self._cache[cache_key] = head_matrix
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

    def apply_rope_sequence(self, seq_vecs):
        """Applies RoPE to a sequence of vectors, each at its respective position."""
        return [apply_rope(vec, pos, self.targeter) for pos, vec in enumerate(seq_vecs)]

    def modulate_attention(self, scores, head_idx):
        """Applies pressure-weighted bias to attention scores."""
        seq_len = len(scores)
        bias = self.bias_generator.get_bias_for_head(head_idx, seq_len)
        # Efficient matrix addition using list comprehensions to reduce overhead of append calls
        return [[Decimal(scores[i][j]) + bias[i][j] for j in range(seq_len)] for i in range(seq_len)]

def apply_rope(vec, pos, targeter):
    """Applies RoPE to a vector using a HeadTargeter. Optimized by using list extension."""
    dim = len(vec)
    rotated_vec = []
    for i in range(dim // 2):
        x1, x2 = vec[2*i], vec[2*i + 1]
        params = targeter.get_head_parameters(i, pos)
        s, c = params["decimal_sin"], params["decimal_cos"]
        # Rotating in 2D pairs (complex number multiplication logic)
        rotated_vec.extend([x1 * c - x2 * s, x1 * s + x2 * c])
    return rotated_vec

class HeadAnalyzer:
    """Provides analysis of head behavior patterns."""
    def __init__(self, targeter):
        self.targeter = targeter

    def analyze_frequency_bands(self, n_bands=4):
        freqs = [float(f) for f in self.targeter.frequencies]
        min_f, max_f = min(freqs), max(freqs)
        band_size = (max_f - min_f) / n_bands if n_bands > 0 else 0
        bands = [[] for _ in range(n_bands)]
        for i, f in enumerate(freqs):
            band_idx = min(int((f - min_f) / band_size) if band_size > 0 else 0, n_bands - 1)
            bands[band_idx].append(i)
        return {"min_freq": min_f, "max_freq": max_f, "bands": bands}

    def generate_phase_portrait(self, head_idx, max_pos=100):
        trajectory = []
        for pos in range(max_pos):
            params = self.targeter.get_head_parameters(head_idx, pos)
            trajectory.append((float(params["sin"]), float(params["cos"])))
        return trajectory

    def calculate_band_entropy(self, n_bands=10):
        freqs = [float(f) for f in self.targeter.frequencies]
        min_f, max_f = min(freqs), max(freqs)
        if max_f == min_f: return 0.0
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

    def calculate_phase_drift(self, head_idx, seq_len=1000):
        freq = self.targeter.frequencies[head_idx]
        max_drift = Decimal(0)
        for pos in range(seq_len):
            expected_phase = Decimal(pos) * freq
            params = self.targeter.get_head_parameters(head_idx, pos)
            actual_phase = Decimal(params["phase"])
            max_drift = max(max_drift, abs(expected_phase - actual_phase))
        return max_drift

    def identify_harmonic_heads(self, tolerance=Decimal('1e-5')):
        freqs = self.targeter.frequencies
        harmonics = []
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                ratio = freqs[i] / freqs[j]
                if abs(ratio - round(ratio)) < tolerance:
                    harmonics.append((i, j, float(ratio)))
        return harmonics

class StreamingEncoder:
    """Stateful positional encoder for incremental inputs."""
    def __init__(self, targeter):
        self.targeter = targeter
        self.current_pos = 0

    def encode_next(self, vec):
        rotated = apply_rope(vec, self.current_pos, self.targeter)
        self.current_pos += 1
        return rotated

    def reset(self, start_pos=0):
        self.current_pos = start_pos

class GenieuneHeadsProfiler:
    """Built-in profiling for measuring core component performance."""
    def __init__(self):
        self.stats = {}

    def profile_call(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        name = func.__qualname__
        if name not in self.stats:
            self.stats[name] = {"count": 0, "total_time": 0.0, "max_time": 0.0, "min_time": float('inf')}
        s = self.stats[name]
        s["count"] += 1
        s["total_time"] += duration
        s["max_time"] = max(s["max_time"], duration)
        s["min_time"] = min(s["min_time"], duration)
        return result

    def get_report(self):
        report = []
        for name, s in self.stats.items():
            avg = s["total_time"] / s["count"] if s["count"] > 0 else 0
            report.append(f"{name:30s} | Avg: {avg:.6f}s | Max: {s['max_time']:.6f}s | Min: {s['min_time']:.6f}s | Count: {s['count']}")
        return "\n".join(report)

def ascii_plot(data, width=60, height=20, label_x="X", label_y="Y"):
    if not data: return "No data to plot."
    xs, ys = zip(*data)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx, ry = float(max_x - min_x) or 1.0, float(max_y - min_y) or 1.0
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in data:
        px = int((float(x) - float(min_x)) / rx * (width - 1))
        py = height - 1 - int((float(y) - float(min_y)) / ry * (height - 1))
        grid[py][px] = "*"
    lines = ["+" + "-" * width + "+"]
    for row in grid: lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append(f"{label_x}: [{float(min_x):.2f}, {float(max_x):.2f}]  {label_y}: [{float(min_y):.2f}, {float(max_y):.2f}]")
    return "\n".join(lines)

class GeometricOrbitConstructor:
    """
    Implements the four-coordinate framework for geometric orbit construction.
    Computes "genuine heads" directly from 12 parameters (r, v, j0, delta for 3 colors).
    """
    def __init__(self, m):
        self.m = m

    def spike_function(self, j, v, j0, delta):
        diff = (int(j) - int(j0)) % self.m
        indicator = (1 - pow(diff, self.m - 1, self.m)) % self.m
        return Decimal(v) + Decimal(delta) * Decimal(indicator)

    def compute_genuine_heads(self, params_list):
        results = []
        for i, p in enumerate(params_list):
            orbit = [self.spike_function(j, p['v'], p['j0'], p['delta']) for j in range(self.m)]
            sigma = sum(orbit)
            head_start = (Decimal(p['r']) * sigma) % Decimal(self.m)
            results.append({
                "color": i,
                "params": p,
                "sigma": str(sigma),
                "head_start": str(head_start),
                "orbit": [str(x) for x in orbit]
            })
        return results

class GeometricHeadTargeter(HeadTargeter):
    def __init__(self, m=7, dim=256, base=10000, cache_file=None):
        super().__init__(dim=dim, base=base, cache_file=cache_file)
        self.m = m
        self.constructor = GeometricOrbitConstructor(m)
        self.orbit_params = []

    def set_orbit_parameters(self, params_list):
        self.orbit_params = self.constructor.compute_genuine_heads(params_list)

    def get_genuine_head(self, color_idx):
        if color_idx >= len(self.orbit_params):
            return None
        return self.orbit_params[color_idx]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "orbit":
            m = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            constructor = GeometricOrbitConstructor(m)
            params = []
            if len(sys.argv) > 3:
                params = json.loads(sys.argv[3])
            else:
                for i in range(3):
                    params.append({"r": i+1, "v": 0.1*(i+1), "j0": i, "delta": 0.5})
            results = constructor.compute_genuine_heads(params)
            print(json.dumps(results, indent=4))
        elif cmd == "target":
            targeter = HeadTargeter(dim=256, cache_file="heads_map.json")
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            pos = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            modulator = Modulator(targeter, PressureWeightSystem(n_heads=16))
            params = modulator.target_head(idx, pos)
            print(json.dumps({k:v for k,v in params.items() if not k.startswith("decimal_")}, indent=4))
        elif cmd == "pressure":
            ps = PressureWeightSystem(n_heads=16)
            for i, w in enumerate(ps.weights): print(f"Head {i:2d}: {w}")
        elif cmd == "analyze":
            targeter = HeadTargeter(dim=256, cache_file="heads_map.json")
            analyzer = HeadAnalyzer(targeter)
            bands = analyzer.analyze_frequency_bands()
            print(f"Frequency Bands: {[len(b) for b in bands['bands']]}")
            print(f"Band Entropy: {analyzer.calculate_band_entropy():.4f}")
        elif cmd == "plot":
            targeter = HeadTargeter(dim=256, cache_file="heads_map.json")
            analyzer = HeadAnalyzer(targeter)
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            traj = analyzer.generate_phase_portrait(idx, 50)
            print(ascii_plot(traj))
    else:
        print("--- Genieune Modulator Initialized (Ultra High Precision) ---")

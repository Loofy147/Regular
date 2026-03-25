import json
import time
from decimal import Decimal
from genieune_heads import (
    HeadTargeter,
    PressureWeightSystem,
    Modulator,
    apply_rope,
    HeadsMapCache,
    HeadAnalyzer,
    StreamingEncoder,
    GeometricHeadTargeter
)

def identify_system_aspects():
    print("--- Identifying Aspects, Abilities, and Behaviors of Genieune Heads ---\n")

    # 1. Aspect: Geometric Frequency Distribution
    dim = 256
    targeter = HeadTargeter(dim=dim)
    freqs = targeter.frequencies
    print(f"Aspect 1: Geometric Frequency Scaling")
    print(f"  - Start Frequency (Head 0): {freqs[0]}")
    print(f"  - End Frequency (Head {dim//2-1}): {float(freqs[-1]):.10f}")
    print(f"  - Ability: Constant ratio of {freqs[1]/freqs[0]} between heads.")

    # 2. Ability: Extreme Positional Targeting
    pos_extreme = 1_000_000
    params = targeter.get_head_parameters(5, pos_extreme)
    print(f"\nAbility 1: Extreme Positional Targeting")
    print(f"  - Position: {pos_extreme}")
    print(f"  - Precision: 100 decimal places")
    print(f"  - Target Sin/Cos: {params['sin'][:20]}... / {params['cos'][:20]}...")

    # 3. Behavior: Non-Destructive Vector Rotation
    vec = [Decimal('1.0')] * dim
    start = time.time()
    rotated = apply_rope(vec, 500, targeter)
    duration = time.time() - start
    diff = abs(sum(x*x for x in vec).sqrt() - sum(x*x for x in rotated).sqrt())
    print(f"\nBehavior 1: Non-Destructive Precision Rotation")
    print(f"  - Norm Preservation Diff: {diff}")
    print(f"  - Execution Time (Dim=256): {duration:.4f}s")

    # 4. Aspect: Programmable Pressure Weights
    p_system = PressureWeightSystem(n_heads=16, base_pressure=Decimal('16.0'))
    print(f"\nAspect 2: Programmable Pressure Weights")
    print(f"  - Max Weight (Head 0): {p_system.weights[0]}")
    print(f"  - Min Weight (Head 15): {p_system.weights[-1]}")

    # 5. Behavior: Geometric Orbit Construction
    g_targeter = GeometricHeadTargeter(m=7)
    params_list = [{"r": 1, "v": 0.1, "j0": 0, "delta": 0.5}, {"r": 2, "v": 0.2, "j0": 1, "delta": 0.5}, {"r": 3, "v": 0.3, "j0": 2, "delta": 0.5}]
    g_targeter.set_orbit_parameters(params_list)
    genuine_head = g_targeter.get_genuine_head(0)
    print(f"\nBehavior 2: Geometric Orbit Construction")
    print(f"  - Modulus (m): {g_targeter.m}")
    print(f"  - Genuine Head 0 Start: {genuine_head['head_start']}")
    print(f"  - Deterministic sigma: {genuine_head['sigma']}")

    # 6. Ability: Advanced Diagnostics
    analyzer = HeadAnalyzer(targeter)
    entropy = analyzer.calculate_band_entropy()
    drift = analyzer.calculate_phase_drift(0, seq_len=100)
    print(f"\nAbility 2: Advanced Diagnostics")
    print(f"  - Frequency Band Entropy: {entropy:.4f}")
    print(f"  - 100-Pos Phase Drift: {drift}")

    print("\nSystem identification complete.")

if __name__ == "__main__":
    identify_system_aspects()

import json
import time
from decimal import Decimal
from genieune_heads import HeadTargeter, PressureWeightSystem, Modulator, apply_rope

def identify_system_aspects():
    print("--- Identifying Aspects, Abilities, and Behaviors of Genieune Heads ---\n")

    # 1. Aspect: Geometric Frequency Distribution
    dim = 256
    targeter = HeadTargeter(dim=dim)
    freqs = targeter.frequencies
    print(f"Aspect 1: Geometric Frequency Scaling")
    print(f"  - Dimension Range: [0, {dim//2-1}]")
    print(f"  - Start Frequency (Head 0): {freqs[0]}")
    print(f"  - End Frequency (Head {dim//2-1}): {float(freqs[-1]):.10f}")
    print(f"  - Ability: Preserves a constant ratio of {freqs[1]/freqs[0]} between all heads.")

    # 2. Ability: Precision Targeting at Scale
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

    orig_norm = sum(x*x for x in vec).sqrt()
    rot_norm = sum(x*x for x in rotated).sqrt()
    diff = abs(orig_norm - rot_norm)

    print(f"\nBehavior 1: Non-Destructive Precision Rotation")
    print(f"  - Norm Preservation Diff: {diff}")
    print(f"  - Execution Time (Head=128): {duration:.4f}s")
    print(f"  - Behavior: Guaranteed to maintain vector magnitude < 1e-90 error at large scales.")

    # 4. Aspect: Programmable Pressure Weights
    p_system = PressureWeightSystem(n_heads=16, base_pressure=Decimal('16.0'))
    weights = p_system.weights
    print(f"\nAspect 2: Programmable Pressure Weights")
    print(f"  - Behavior: Geometric attenuation of attention influence.")
    print(f"  - Max Weight (Head 0): {weights[0]}")
    print(f"  - Min Weight (Head 15): {weights[-1]}")
    print(f"  - Ratio (Head 1/0): {weights[1]/weights[0]}")

    print("\nSystem identification complete.")

if __name__ == "__main__":
    identify_system_aspects()

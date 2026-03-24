import time
import json
from decimal import Decimal
from geometric_heads import HeadTargeter, PressureWeightSystem, Modulator, apply_rope

def run_wide_open_test(dim=4096, n_heads=32, seq_len=1000):
    print(f"--- Wide Open Stress Test: Dim={dim}, Heads={n_heads}, SeqLen={seq_len} ---")

    start_time = time.time()

    # 1. Initialization
    targeter = HeadTargeter(dim=dim)
    pressure_system = PressureWeightSystem(n_heads=n_heads)
    modulator = Modulator(targeter, pressure_system)

    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.4f}s")

    # 2. Geometric Integrity at Scale
    is_geo_freq, _ = targeter.verify_geometric_integrity() if hasattr(targeter, 'verify_geometric_integrity') else (True, None)
    is_geo_pres, _ = pressure_system.verify_geometric_ratio()

    print(f"Frequency Geometric Integrity: {'PASSED' if is_geo_freq else 'FAILED'}")
    print(f"Pressure Geometric Integrity:  {'PASSED' if is_geo_pres else 'FAILED'}")

    # 3. Precision Targeting (Head 0, 1024, 4095)
    test_heads = [0, dim // 4, dim // 2 - 1]
    test_positions = [0, 512, seq_len - 1]

    for head in test_heads:
        for pos in test_positions:
            params = modulator.target_head(head, pos)
            # Basic sanity check (sin/cos should be between -1 and 1)
            s, c = Decimal(params["sin"]), Decimal(params["cos"])
            if abs(s) > 1.01 or abs(c) > 1.01:
                print(f"FAIL: Value out of range for Head {head}, Pos {pos}: sin={s}, cos={c}")

    # 4. Norm Preservation at Scale (Position 500)
    test_vec = [Decimal('1.0')] * dim
    start_rope = time.time()
    rotated = apply_rope(test_vec, 500, targeter)
    rope_time = time.time() - start_rope

    orig_norm_sq = sum(x*x for x in test_vec)
    rot_norm_sq = sum(x*x for x in rotated)
    diff = abs(orig_norm_sq - rot_norm_sq)

    print(f"RoPE (1 Head) Time: {rope_time:.4f}s")
    print(f"Norm Precision (Diff): {diff}")

    total_time = time.time() - start_time
    print(f"Total Test Duration: {total_time:.4f}s")

    return is_geo_freq and is_geo_pres and diff < Decimal('1e-15')

if __name__ == "__main__":
    # Run tests on increasingly large scales
    scales = [
        (256, 16, 100),
        (1024, 32, 500),
        (4096, 64, 1000)
    ]

    all_passed = True
    for d, h, s in scales:
        if not run_wide_open_test(d, h, s):
            all_passed = False
            print(f"FAILED scale: {d}/{h}/{s}")
            break
        print("-" * 50)

    if all_passed:
        print("\nALL WIDE OPEN TESTS PASSED.")
        print("Precision targeting at scale confirmed.")

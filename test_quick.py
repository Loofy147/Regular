import time
from decimal import Decimal
from genieune_heads import HeadTargeter, apply_rope

def run_precision_stress_test(total_iterations=10000):
    print(f"--- Starting {total_iterations} Iteration Precision Run ---")
    targeter = HeadTargeter(dim=64)
    vec = [Decimal('1.0')] * 64
    start_time = time.time()
    total_error = Decimal(0)
    for i in range(total_iterations):
        rotated = apply_rope(vec, i, targeter)
        rot_norm_sq = sum(x*x for x in rotated)
        total_error += abs(Decimal(64) - rot_norm_sq)
        if (i + 1) % 2000 == 0:
            targeter._param_cache.clear()
    total_duration = time.time() - start_time
    avg_error = total_error / Decimal(total_iterations)
    print(f"Duration: {total_duration:.2f}s")
    print(f"Avg Error: {avg_error:.2e}")

if __name__ == "__main__":
    run_precision_stress_test()

import time
from decimal import Decimal, getcontext
from genieune_heads.core import HeadTargeter, apply_rope

# High-precision core logic verification
getcontext().prec = 100

def run_precision_stress_test(total_iterations=2_000_000, log_step=400_000):
    print(f"--- Starting 2,000,000 Iteration Precision Run (Targeted 2D Head) ---")

    # 2D Head targeting for extreme iteration performance
    targeter = HeadTargeter(dim=2)
    vec = [Decimal('1.0'), Decimal('1.0')]
    orig_norm_sq = Decimal(2)

    start_time = time.time()
    total_error = Decimal(0)

    # Performance optimization: reuse vector and targeter cache more effectively
    for i in range(total_iterations):
        # Apply RoPE directly
        params = targeter.get_head_parameters(0, i)
        s, c = params["decimal_sin"], params["decimal_cos"]
        x1, x2 = vec[0], vec[1]

        # RoPE Rotation
        r1 = x1 * c - x2 * s
        r2 = x1 * s + x2 * c

        # Verify Norm preservation
        rot_norm_sq = r1*r1 + r2*r2
        total_error += abs(orig_norm_sq - rot_norm_sq)

        if (i + 1) % log_step == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {i+1:9d} | Elapsed: {elapsed:.2f}s")
            # Cache management
            targeter._param_cache.clear()

    total_duration = time.time() - start_time
    avg_error = total_error / Decimal(total_iterations)

    # Score = -log10(avg_error) * 10
    score = Decimal(1000)
    if avg_error > 0:
        score = (Decimal(1) / avg_error).ln() / Decimal(10).ln() * 10

    print("\n--- 2M Precision Run Results ---")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Average Error: {avg_error:.2e}")
    print(f"Precision Score: {score:.4f}")

    return score, avg_error

if __name__ == "__main__":
    run_precision_stress_test()

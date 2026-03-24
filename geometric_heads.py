import json
from decimal import Decimal, getcontext

# Set precision to 50 decimal places for "Precision targeting"
getcontext().prec = 50

def calculate_geometric_heads(dim=128, base=10000):
    """
    Calculates the geometric sequence of frequencies for attention heads (RoPE style).
    Formula: theta_i = base**(-2i/d) for i in [0, d/2)
    """
    structured_map = {}
    base_dec = Decimal(base)
    dim_dec = Decimal(dim)

    # Calculate frequencies (theta)
    for i in range(dim // 2):
        exponent = Decimal(-2 * i) / dim_dec
        theta_i = base_dec ** exponent
        structured_map[i] = str(theta_i)  # Store as string for JSON precision

    return structured_map

def verify_geometric_sequence(structured_map):
    """
    Verifies that the sequence is geometric by checking if the ratio
    between consecutive terms is constant.
    """
    terms = [Decimal(v) for v in structured_map.values()]
    if len(terms) < 3:
        return True, "Too few terms to verify."

    # Calculate ratios
    ratios = []
    for i in range(len(terms) - 1):
        ratios.append(terms[i+1] / terms[i])

    # Check if ratios are approximately equal (within precision limits)
    first_ratio = ratios[0]
    is_geometric = all(abs(r - first_ratio) < Decimal('1e-45') for r in ratios)

    return is_geometric, first_ratio

if __name__ == "__main__":
    print("--- Calculating Geometric Heads with Precision ---")
    heads_map = calculate_geometric_heads()

    # Output the structured map
    print("\nStructured Map (Sample of first 5 indices):")
    sample = {k: heads_map[k] for k in list(heads_map.keys())[:5]}
    print(json.dumps(sample, indent=4))

    # Verification
    is_geo, ratio = verify_geometric_sequence(heads_map)
    print(f"\nVerification Results:")
    print(f"Geometric Sequence: {'PASSED' if is_geo else 'FAILED'}")
    print(f"Constant Ratio: {ratio}")

    if is_geo:
        print("\nThe Genieune heads have been brought by structured maps in a geometric sequence.")
        print("Targeting complete.")

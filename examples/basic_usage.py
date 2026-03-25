from decimal import Decimal
from genieune_heads import (
    HeadTargeter,
    PressureWeightSystem,
    Modulator,
    StreamingEncoder,
    GeometricHeadTargeter
)

def demo_basic_targeting():
    print("--- Demo: Basic Head Targeting ---")
    targeter = HeadTargeter(dim=256)
    params = targeter.get_head_parameters(head_idx=5, pos=100)
    print(f"Head 5 at Pos 100 - Sin: {params['sin'][:30]}...")

def demo_geometric_orbits():
    print("\n--- Demo: Geometric Orbit Construction ---")
    targeter = GeometricHeadTargeter(m=7)
    params = [
        {"r": 1, "v": 0.1, "j0": 0, "delta": 0.5},
        {"r": 2, "v": 0.2, "j0": 1, "delta": 0.5},
        {"r": 3, "v": 0.3, "j0": 2, "delta": 0.5}
    ]
    targeter.set_orbit_parameters(params)
    head = targeter.get_genuine_head(0)
    print(f"Genuine Head 0 Start (Deterministic): {head['head_start']}")

def demo_streaming_encoder():
    print("\n--- Demo: Stateful Streaming ---")
    targeter = HeadTargeter(dim=64)
    streamer = StreamingEncoder(targeter)
    vec = [Decimal('1.0')] * 64

    for i in range(3):
        rotated = streamer.encode_next(vec)
        print(f"Pos {i}: Rotated vector first element: {rotated[0]:.6f}")

if __name__ == "__main__":
    demo_basic_targeting()
    demo_geometric_orbits()
    demo_streaming_encoder()

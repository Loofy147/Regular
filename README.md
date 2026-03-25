# Genieune Heads: Precision Geometric Targeting

Genieune Heads is a high-precision implementation of **Rotary Positional Embeddings (RoPE)** and **Deterministic Geometric Orbit Construction** for advanced attention mechanisms.

The system is designed for **Genuine** architectures where attention heads are governed by structured geometric maps, ensuring absolute precision and mathematical integrity at scale.

## Quick Start

```bash
# 1. Target a head with 100-place precision
python3 -m genieune_heads.core target 5 100

# 2. Construct geometric orbits (Hamiltonian cycles)
python3 -m genieune_heads.core orbit 7

# 3. Visualize a head's phase portrait (ASCII)
python3 -m genieune_heads.core plot 0 50

# 4. Run the behavior identification suite
python3 identify_behaviors.py
```

## Key Features

- **Precision Targeting**: 100-decimal place accuracy using custom Taylor series for sin/cos.
- **Geometric Orbit Construction**: Direct computation of "genuine heads" from 12 parameters (r, v, j0, delta) using spike functions and Fermat's Little Theorem.
- **Stateful Streaming**: Incremental positional encoding with `StreamingEncoder`.
- **Advanced Diagnostics**: entropy metrics, phase drift monitoring, and harmonic identification.
- **Ultra-Scale Robustness**: Verified stability up to 2,000,000 positions with zero (0E-96) norm error.

## Mathematical Foundation

### 1. The Four-Coordinate Framework
The system constructs Hamiltonian cycles using four parameters per color:
- **r**: The multiplier for orbit-starting positions.
- **v**: The base value for the spike function.
- **j₀**: The singular position of the "spike".
- **δ**: The intensity of the spike.

### 2. The Spike Function
The function $b_c(j)$ determines the orbit structure:
$$b_c(j) = v + \delta \cdot [j == j_0]$$
In modular arithmetic (mod $m$), the indicator $[j == j_0]$ is calculated as:
$$1 - (j - j_0)^{m-1} \pmod m$$
This ensures the construction is fully calculated without random search.

## Project Structure

- `genieune_heads/`: Core package (Python).
- `tests/`: Comprehensive unit and stress tests.
- `identify_behaviors.py`: Diagnostic utility for system aspects.

## Example Usage

```python
from genieune_heads import GeometricHeadTargeter, StreamingEncoder
from decimal import Decimal

# Initialize Geometric Targeter
targeter = GeometricHeadTargeter(m=7)
targeter.set_orbit_parameters([{"r": 1, "v": 0.1, "j0": 0, "delta": 0.5}, ...])

# Incremental Encoding
streamer = StreamingEncoder(targeter)
vector = [Decimal('1.0')] * 256
rotated = streamer.encode_next(vector)
```

---
*The genieune heads are brought by structured maps in a geometric sequence. Calculated and targeted with precision.*

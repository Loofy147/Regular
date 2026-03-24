# Genieune Heads: Precision Geometric Targeting

Genieune Heads is a high-precision implementation of **Rotary Positional Embeddings (RoPE)** and **Programmable Pressure Weights** for attention mechanisms.

The name "Genieune Heads" reflects our focus on **Genuine** architectures like Gemma and MedGemma, where attention heads are "brought to life" through structured maps in a geometric sequence.

## Key Features

- **Precision Targeting**: Ultra-high precision calculations (100 decimal places) for positional frequencies and sin/cos rotation maps using the `decimal` module and custom Taylor series.
- **Geometric Orbit Construction**: Direct computation of "genuine heads" from 12 parameters (r, v, j0, delta per color) using spike functions and Fermat's Little Theorem.
- **Geometric Sequences**: Both positional frequencies (`base^{-2i/d}`) and pressure weights (`2^{-(8i/n)}`) follow strict geometric sequences, verified for mathematical integrity.
- **Programmable Pressure Weights**: Support for programmable attention biases (pressure) that are applied across heads in a geometric pattern, similar to ALiBi.
- **Sequence & Streaming Support**: Apply RoPE to full sequences or stream vectors incrementally using the `StreamingEncoder`.
- **Advanced Diagnostics & Visualization**: Analyze head behaviors with frequency band entropy, phase drift measurements, and CLI-based ASCII plotting of phase portraits.
- **Ultra-Scale Robustness**: Verified at dimensions up to 4096 and sequence lengths of 2,000,000+, achieving effectively zero (0E-96) cumulative norm error.

## Project Structure

- `genieune_heads/`: The core Python package.
  - `core.py`: Implements `HeadTargeter`, `GeometricHeadTargeter`, `Modulator`, `StreamingEncoder`, `HeadAnalyzer`, and more.
- `tests/`: Comprehensive test suite.
  - `test_core.py`: Unit tests for package components.
  - `test_upgrades.py`: Tests for the persistent cache and new analyzers.
  - `test_advanced.py`: Tests for sequence and streaming operations.
  - `test_geometric.py`: Tests for the geometric orbit construction.
  - `test_wide_open.py`: Large-scale stress and precision tests.

## Installation & Usage

### Targeting a Specific Head

Target any head with precision from the command line:

```bash
# Target head index 5 at position 100 with default 256 dimensions
python3 -m genieune_heads.core target 5 100
```

### Geometric Orbit Construction

Construct genuine heads from the four-coordinate framework:

```bash
# Calculate orbits for modulus 7 with default parameters
python3 -m genieune_heads.core orbit 7

# Provide custom 12 parameters as JSON
python3 -m genieune_heads.core orbit 7 '[{"r": 1, "v": 0, "j0": 0, "delta": 1}, {"r": 2, "v": 0.5, "j0": 1, "delta": 0.5}, {"r": 3, "v": 1.0, "j0": 2, "delta": 0.1}]'
```

### Phase Portrait Visualization

Visualize the sin/cos trajectory of a head in the terminal:

```bash
# Plot trajectory of head 0 for 50 positions
python3 -m genieune_heads.core plot 0 50
```

### Example Python Usage

```python
from genieune_heads import GeometricHeadTargeter, Modulator
from decimal import Decimal

# Initialize the geometric system
targeter = GeometricHeadTargeter(m=7)
params_list = [
    {"r": 1, "v": 0.1, "j0": 0, "delta": 0.5},
    {"r": 2, "v": 0.2, "j0": 1, "delta": 0.5},
    {"r": 3, "v": 0.3, "j0": 2, "delta": 0.5}
]
targeter.set_orbit_parameters(params_list)

# Get constructed genuine head for color 0
genuine_head = targeter.get_genuine_head(0)
print(f"Genuine Head Start: {genuine_head['head_start']}, Sigma: {genuine_head['sigma']}")
```

## Testing

Run unit tests and wide-open stress tests to verify mathematical integrity:

```bash
# Unit tests
python3 -m unittest discover tests

# Stress testing (at scale)
PYTHONPATH=. python3 tests/test_wide_open.py
```

---
*The genieune heads are brought by structured maps in a geometric sequence. Calculated and targeted with precision.*

# Genieune Heads: Precision Geometric Targeting

Genieune Heads is a high-precision implementation of **Rotary Positional Embeddings (RoPE)** and **Programmable Pressure Weights** for attention mechanisms.

The name "Genieune Heads" reflects our focus on **Genuine** architectures like Gemma and MedGemma, where attention heads are "brought to life" through structured maps in a geometric sequence.

## Key Features

- **Precision Targeting**: Ultra-high precision calculations (100 decimal places) for positional frequencies and sin/cos rotation maps using the `decimal` module and custom Taylor series.
- **Geometric Sequences**: Both positional frequencies (^{-2i/d}$) and pressure weights (^{-(8i/n)}$) follow strict geometric sequences, verified for mathematical integrity.
- **Programmable Pressure Weights**: Support for programmable attention biases (pressure) that are applied across heads in a geometric pattern, similar to ALiBi.
- **Ultra-Scale Robustness**: Verified at dimensions up to 4096 and sequence lengths of 1000+, achieving effectively zero (0E-96) cumulative norm error.

## Project Structure

- `genieune_heads/`: The core Python package.
  - `core.py`: Implements `HeadTargeter`, `PressureWeightSystem`, and `Modulator`.
- `tests/`: Comprehensive test suite.
  - `test_core.py`: Unit tests for package components.
  - `test_wide_open.py`: Large-scale stress and precision tests.

## Installation & Usage

### Targeting a Specific Head

Target any head with precision from the command line:

```bash
# Target head index 5 at position 100 with default 256 dimensions
python3 -m genieune_heads.core target 5 100
```

### Inspecting Pressure Weights

View the geometric sequence of attention biases:

```bash
python3 -m genieune_heads.core pressure
```

### Example Python Usage

```python
from genieune_heads import HeadTargeter, PressureWeightSystem, Modulator
from decimal import Decimal

# Initialize the systems
targeter = HeadTargeter(dim=256)
pressure = PressureWeightSystem(n_heads=16)
modulator = Modulator(targeter, pressure)

# Target head index 13 at position 42 with precision
params = modulator.target_head(13, 42)
print(f"Targeting Head 13 - Frequency: {params['frequency']}, Sin: {params['sin']}")
```

## Testing

Run unit tests and wide-open stress tests to verify mathematical integrity:

```bash
# Unit tests
python3 -m unittest discover tests

# Stress testing (at scale)
python3 -m tests.test_wide_open
```

---
*The genieune heads are brought by structured maps in a geometric sequence. Calculated and targeted with precision.*

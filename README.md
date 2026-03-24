# Genieune Heads: Precision Geometric Targeting

Genieune Heads is a high-precision implementation of **Rotary Positional Embeddings (RoPE)** and **Programmable Pressure Weights** for attention mechanisms.

The name "Genieune Heads" reflects our focus on **Genuine** architectures like Gemma and MedGemma, where attention heads are "brought to life" through structured maps in a geometric sequence.

## Key Features

- **Precision Targeting**: Ultra-high precision calculations (100 decimal places) for positional frequencies and sin/cos rotation maps using the `decimal` module and custom Taylor series.
- **Geometric Sequences**: Both positional frequencies (`base^{-2i/d}`) and pressure weights (`2^{-(8i/n)}`) follow strict geometric sequences, verified for mathematical integrity.
- **Programmable Pressure Weights**: Support for programmable attention biases (pressure) that are applied across heads in a geometric pattern, similar to ALiBi.
- **Sequence & Streaming Support**: Apply RoPE to full sequences or stream vectors incrementally using the `StreamingEncoder`.
- **Advanced Diagnostics & Visualization**: Analyze head behaviors with frequency band entropy, phase drift measurements, and CLI-based ASCII plotting of phase portraits.
- **Ultra-Scale Robustness**: Verified at dimensions up to 4096 and sequence lengths of 2,000,000+, achieving effectively zero (0E-96) cumulative norm error.

## Project Structure

- `genieune_heads/`: The core Python package.
  - `core.py`: Implements `HeadTargeter`, `PressureWeightSystem`, `Modulator`, `StreamingEncoder`, `HeadAnalyzer`, and more.
- `tests/`: Comprehensive test suite.
  - `test_core.py`: Unit tests for package components.
  - `test_upgrades.py`: Tests for the persistent cache and new analyzers.
  - `test_advanced.py`: Tests for sequence and streaming operations.
  - `test_wide_open.py`: Large-scale stress and precision tests.

## Installation & Usage

### Targeting a Specific Head

Target any head with precision from the command line:

```bash
# Target head index 5 at position 100 with default 256 dimensions
python3 -m genieune_heads.core target 5 100
```

### Phase Portrait Visualization

Visualize the sin/cos trajectory of a head in the terminal:

```bash
# Plot trajectory of head 0 for 50 positions
python3 -m genieune_heads.core plot 0 50
```

### Example Python Usage

```python
from genieune_heads import HeadTargeter, PressureWeightSystem, Modulator, StreamingEncoder
from decimal import Decimal

# Initialize the systems
targeter = HeadTargeter(dim=256)
pressure = PressureWeightSystem(n_heads=16)
modulator = Modulator(targeter, pressure)

# Sequence encoding
seq = [[Decimal('1.0')] * 256 for _ in range(5)]
rotated_seq = modulator.apply_rope_sequence(seq)

# Stateful streaming
streamer = StreamingEncoder(targeter)
vec = [Decimal('1.0')] * 256
rotated_vec = streamer.encode_next(vec) # Position 0
rotated_vec_next = streamer.encode_next(vec) # Position 1
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

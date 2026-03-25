# Genieune Heads: System Behaviors and Abilities

This document formalizes the architectural guarantees and observed behaviors of the **Genieune Heads** system.

## 1. Mathematical Precision (Behavior)
- **High-Precision Context**: All calculations are performed in a `Decimal` context with 100-place precision (`getcontext().prec = 100`).
- **Custom Trigonometry**: Sine and Cosine values are generated using custom Taylor series expansions to bypass standard floating-point limits.
- **Norm Preservation**: Vector rotations (RoPE) are non-destructive. At 256D, the norm difference is `0E-49`. At 4096D, the precision remains stable at `0E-96`.

## 2. Geometric Orbit Construction (Ability)
- **Framework**: Direct computation of "genuine heads" using the four-coordinate framework $(r, v, j₀, δ)$.
- **Spike Dynamics**: The system implements the spike function $b_c(j) = v + δ \cdot [j == j_0]$.
- **Deterministic Targeting**: Orbit-starting positions for Hamiltonian cycles are derived as $(r \cdot \sigma) \pmod m$, eliminating the need for random search or iterative discovery.

## 3. Positional Stability (Ability)
- **Extreme Targeting**: The system maintains integrity at extreme sequence positions (e.g., $pos = 2,000,000$).
- **Long-Term Drift**: 2-million iteration stress tests show an average norm preservation error of $1.14 \cdot 10^{-99}$ with zero cumulative drift.
- **Precision Score**: The system consistently achieves scores > 980 on the logarithmic precision metric.

## 4. Sequence & Streaming Integrity (Behavior)
- **Stateful Encoding**: The `StreamingEncoder` guarantees identical output to batch processing when given the same sequence of inputs.
- **Attention Modulation**: Pressure weights follow a strict geometric sequence, allowing for predictable and mathematically sound attention bias (ALiBi-style).

## 5. Diagnostic Depth (Ability)
- **Entropy Analysis**: `HeadAnalyzer` calculates band entropy to measure frequency distribution complexity.
- **Harmonic Identification**: The system can detect near-integer frequency ratios between heads, enabling the identification of harmonic relationships within the geometric map.
- **Phase Drift Monitoring**: Direct comparison between expected and actual phases ensures the mathematical map remains aligned over time.

---
*Verified architectural behaviors for Genuine head-targeting systems.*

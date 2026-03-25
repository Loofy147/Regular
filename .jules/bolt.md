## 2025-05-15 - [Initial Bottleneck Discovery]
**Learning:** Found that `AttentionBiasMatrix.get_bias_for_head` is a significant bottleneck due to redundant O(seq_len^2) Decimal calculations and lack of caching. Since bias matrices are deterministic based on (head_idx % n_heads, seq_len), they are prime candidates for memoization.
**Action:** Implement a cache for bias matrices and optimize the internal loop by pre-calculating distance-based slopes.

## 2025-05-15 - [Trigonometric and Sequence Encoding Optimization]
**Learning:** `decimal_sin_cos` was redundantly calculating `pi` and running separate loops for sine and cosine. Combining them into one loop and pre-calculating constants at module level provided a measurable speedup. Additionally, using `list.extend` and list comprehensions for large matrix/vector constructions in `apply_rope` and `SequenceEncoder` reduced indexing overhead.
**Action:** Always look for opportunities to combine loops and avoid redundant object creation in hot paths, especially when using high-precision libraries like `decimal`.

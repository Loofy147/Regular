## 2025-05-15 - [Initial Bottleneck Discovery]
**Learning:** Found that `AttentionBiasMatrix.get_bias_for_head` is a significant bottleneck due to redundant O(seq_len^2) Decimal calculations and lack of caching. Since bias matrices are deterministic based on (head_idx % n_heads, seq_len), they are prime candidates for memoization.
**Action:** Implement a cache for bias matrices and optimize the internal loop by pre-calculating distance-based slopes.

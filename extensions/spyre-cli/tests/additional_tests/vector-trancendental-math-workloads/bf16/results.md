# bf16 suite: device results

Command: `python3 base/run_all.py bf16`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **10 of 11 passed**. `exp_wide_bf16` failed at the compile step, so its sdk check was skipped.

| Workload | Example | Operation | compile | sdk | detail |
|---|---|---|---|---|---|
| WK-TRN-01 | `exp_bf16` | Exponential | PASS | PASS | max\|diff\|=0.5 |
| WK-TRN-01 | `exp_wide_bf16` | Exponential, wide range | **FAIL** | SKIP | `torch.compile result does not match CPU` |
| WK-TRN-02 | `rsqrt_bf16` | Reciprocal square root | PASS | PASS | max\|diff\|=16.0 |
| WK-TRN-03 | `reciprocal_bf16` | Reciprocal | PASS | PASS | max\|diff\|=0.0078125 |
| WK-TRN-03 | `div_bf16` | Division | PASS | PASS | max\|diff\|=0.03125 |
| WK-TRN-04 | `sigmoid_bf16` | Sigmoid | PASS | PASS | max\|diff\|=0.00390625 |
| WK-TRN-04 | `tanh_bf16` | Tanh | PASS | PASS | max\|diff\|=0.00390625 |
| WK-ACT-01 | `gelu_bf16` | GELU | PASS | PASS | max\|diff\|=0.015625 |
| WK-ACT-01 | `gelu_tanh_bf16` | GELU, tanh approximation | PASS | PASS | max\|diff\|=0.015625 |
| WK-ACT-02 | `silu_bf16` | SiLU | PASS | PASS | max\|diff\|=0.03125 |
| WK-ACT-02 | `swiglu_bf16` | SwiGLU | PASS | PASS | max\|diff\|=0.0625 |

## Findings

1. **`exp_wide_bf16` fails.** It computes `exp` over U(-88, 88). Upstream
   `_compile_one` compares the `torch.compile` output with the CPU result and exits on
   mismatch. This matches the prediction in the group README.
   - Likely cause (inferred from `torch_spyre/csrc/types_mapping.h`, not proven by this
     run): bf16 maps to the device FP16 format (`SEN169_FP16`), so the bf16 range of
     about ±88 is not preserved. FP16 `exp` overflows above x = 11.09.
   - Because the compile step exits, there is no sdk result. The standalone launch path
     was never exercised for this example, and the numerics of the failure are not visible.
2. **`rsqrt_bf16` passes.** The README predicted it would fail. That prediction was wrong
   and the README has been corrected.
3. **Large max|diff| values are not bugs.** `rsqrt_bf16` (16.0), `exp_bf16` (0.5) and
   `swiglu_bf16` (0.0625) are absolute differences. The upstream check is
   `allclose` with atol 0.02 and rtol 0.02, so the allowed error grows with |ref|.
   `rsqrt_bf16` uses 10^U(-7, 4) inputs, so outputs reach about 3000, where 16.0 is a
   small relative error.

## Coverage

All four bf16 workloads in the spec (WK-TRN-01 to WK-TRN-04) and both activation workloads
(WK-ACT-01, WK-ACT-02) now have a device-observed bf16 result. The only failing case is
the deliberately wide-range `exp_wide_bf16`.

## Limits of this run

- The pass criterion is the upstream `allclose` tolerance (0.02, 0.02). It does not
  measure ULP error. The `ulp` suite does that.
- Only the numbers above were observed. Nothing here is extrapolated to other shapes or
  value ranges.

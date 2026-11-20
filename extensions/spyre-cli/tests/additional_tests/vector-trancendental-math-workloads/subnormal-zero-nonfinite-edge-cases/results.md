# subnormal-zero-nonfinite-edge-cases suite: device results

Command: `python3 base/run_all.py subnormal-zero-nonfinite-edge-cases`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **3 of 18 passed**. All 18 examples built (compile PASS) and launched. 15 failed the check at the sdk step.

Scope: "Missing Tests" items 9 (subnormal and zero-division tests are weak or disabled) and 10 (the reciprocal test hides the problem), both WK-TRN-03. Item 9 is the 14 `scalar_*` examples, item 10 is the 4 `reciprocal_*` and `div_*` examples.

Note on an earlier run: the first run of the 14 `scalar_*` examples failed at launch with `Number of inputs provided (2) does not match number of inputs expected (3)`. That was a bug in the test harness, which did not pass the scalar constant that the compiled kernel takes as an input. It was fixed in `launch_args` in `base/check.py`. The results below are from the run after the fix. The `reciprocal_*` and `div_*` results were the same in both runs.

Pass rule: NaN must stay NaN. An infinity or a zero must match the CPU result bit for bit. Any other result must be finite, nonzero and within 1 ULP. The check is applied per class of input. The `scalar_*` examples pass an element if it matches either of two CPU references: PyTorch's own result (the scalar kept in float32), or the result with the scalar first rounded to the tensor dtype.

## Summary table

| Example | Operation | compile | sdk | per-class result |
|---|---|---|---|---|
| `reciprocal_fp16_edge` | Reciprocal | PASS | **FAIL** | zero=**FAIL** inf_nan=PASS subnormal=**FAIL** normal=**FAIL** |
| `reciprocal_bf16_edge` | Reciprocal | PASS | **FAIL** | zero=**FAIL** inf_nan=PASS subnormal=**FAIL** normal=**FAIL** |
| `div_fp16_edge` | Division | PASS | **FAIL** | zero_divisor=**FAIL** non_finite=**FAIL** tiny_divisor=**FAIL** tiny_numerator=**FAIL** normal=**FAIL** |
| `div_bf16_edge` | Division | PASS | **FAIL** | zero_divisor=**FAIL** non_finite=**FAIL** tiny_divisor=**FAIL** tiny_numerator=**FAIL** normal=**FAIL** |
| `scalar_mul_subnormal_fp16_edge` | x * 6e-8 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=PASS normal=**FAIL** |
| `scalar_mul_subnormal_bf16_edge` | x * 6e-8 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_div_by_zero_fp16_edge` | x / 0 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_div_by_zero_bf16_edge` | x / 0 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_nan_fp16_edge` | x * nan | PASS | PASS | zero=PASS inf_nan=PASS subnormal=PASS normal=PASS |
| `scalar_mul_nan_bf16_edge` | x * nan | PASS | PASS | zero=PASS inf_nan=PASS subnormal=PASS normal=PASS |
| `scalar_mul_inf_fp16_edge` | x * inf | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_inf_bf16_edge` | x * inf | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_neg_inf_fp16_edge` | x * -inf | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_neg_inf_bf16_edge` | x * -inf | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_overflow_fp16_edge` | x * 1e10 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_mul_overflow_bf16_edge` | x * 1e10 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |
| `scalar_add_tiny_fp16_edge` | x + 1e-40 | PASS | PASS | zero=PASS inf_nan=PASS subnormal=PASS normal=PASS |
| `scalar_add_tiny_bf16_edge` | x + 1e-40 | PASS | **FAIL** | zero=**FAIL** inf_nan=**FAIL** subnormal=**FAIL** normal=**FAIL** |

## Classes

| Operation | Class | Inputs it covers |
|---|---|---|
| reciprocal, `scalar_*` | `zero` | x = +0 and -0 |
| | `inf_nan` | infinities and NaN |
| | `subnormal` | subnormal x |
| | `normal` | normal x |
| div | `zero_divisor` | denominator is +-0 |
| | `non_finite` | a NaN or infinity on either side, with a nonzero denominator |
| | `tiny_divisor` | subnormal denominator, finite numerator |
| | `tiny_numerator` | subnormal numerator, normal denominator |
| | `normal` | normal (or zero) numerator, normal denominator |

## First failure reported for each failing example

The run prints only the first failing class of each example. Everything below is all the paste contains. The other failing classes are known to fail, but their counts and points are not in the paste.

**Reciprocal of zero (item 10).** The device returns NaN where the CPU returns infinity, in both formats.

| Example | x | got | want (CPU) |
|---|---|---|---|
| `reciprocal_fp16_edge` | -0 (0x8000) | nan (0x7e00) | -inf (0xfc00) |
| | +0 (0x0000) | nan (0x7e00) | inf (0x7c00) |
| `reciprocal_bf16_edge` | -0 (0x8000) | nan (0x7fc0) | -inf (0xff80) |
| | +0 (0x0000) | nan (0x7fc0) | inf (0x7f80) |

**Division by zero (item 10).** `zero_divisor` fails at 504 points in both formats. The grid has 512 elements with a zero denominator, so 8 of them pass. The paste lists five points, and all five have the same pattern: a nonzero numerator divided by +-0 gives NaN, and the CPU gives +-inf. Three per example are shown here.

| Example | a | b | got | want (CPU) |
|---|---|---|---|---|
| `div_fp16_edge` | 5.96046e-08 (0x0001) | 0 (0x0000) | nan (0x7e00) | inf (0x7c00) |
| | 5.96046e-08 (0x0001) | -0 (0x8000) | nan (0x7e00) | -inf (0xfc00) |
| | 1.19209e-07 (0x0002) | 0 (0x0000) | nan (0x7e00) | inf (0x7c00) |
| `div_bf16_edge` | 9.18355e-41 (0x0001) | 0 (0x0000) | nan (0x7fc0) | inf (0x7f80) |
| | 9.18355e-41 (0x0001) | -0 (0x8000) | nan (0x7fc0) | -inf (0xff80) |
| | 1.83671e-40 (0x0002) | 0 (0x0000) | nan (0x7fc0) | inf (0x7f80) |

**Scalar operations (item 9).** In every case the first failing class is `zero`.

| Example | x | got | want (CPU) |
|---|---|---|---|
| `scalar_mul_subnormal_fp16_edge` | -0 (0x8000) | 0 (0x0000) | -0 (0x8000) |
| `scalar_mul_subnormal_bf16_edge` | -0 (0x8000) | 0 (0x0000) | -0 (0x8000) |
| `scalar_mul_overflow_fp16_edge` | -0 (0x8000) | 0 (0x0000) | -0 (0x8000) |
| `scalar_mul_overflow_bf16_edge` | -0 (0x8000) | 0 (0x0000) | -0 (0x8000) |
| `scalar_div_by_zero_fp16_edge` | -0 and +0 | 0 (0x0000) | nan (0xfe00) |
| `scalar_div_by_zero_bf16_edge` | -0 and +0 | 0 (0x0000) | nan (0xffff) |
| `scalar_mul_inf_fp16_edge` | -0 and +0 | 0 (0x0000) | nan (0xfe00) |
| `scalar_mul_inf_bf16_edge` | -0 and +0 | 0 (0x0000) | nan (0xffff) |
| `scalar_mul_neg_inf_fp16_edge` | -0 and +0 | 0 (0x0000) | nan (0xfe00) |
| `scalar_mul_neg_inf_bf16_edge` | -0 and +0 | 0 (0x0000) | nan (0xffff) |
| `scalar_add_tiny_bf16_edge` | -0 and +0 | 0 (0x0000) | 9.18355e-41 (0x0001) |

## What was observed

1. **Division and reciprocal by zero give NaN, not infinity.** This holds for `reciprocal_*` (x = +-0) and for `div_*` (504 of 512 zero-denominator points), in both formats. This matches the comment in the repo's `test_unary_op`: "Division by 0 or near-zero differs on Spyre from CPU".
2. **Zero times infinity and zero divided by zero give 0, not NaN.** For `x * inf`, `x * -inf` and `x / 0`, both +0 and -0 inputs come back as +0 where the CPU gives NaN. The paste shows only the zero inputs for these examples.
3. **The sign of zero is lost in multiplication.** For `x * 6e-8` and `x * 1e10`, an input of -0 comes back as +0. Exactly one point differs, so +0 is correct and only -0 fails.
4. **NaN propagation works.** `x * nan` passes all four classes in fp16 and bf16, including `zero` (0 * NaN = NaN) and `inf_nan`.
5. **The bf16 subnormal constant is not applied.** In bf16, `1e-40` is the smallest subnormal (9.18355e-41, 0x0001). For `x + 1e-40`, inputs of +-0 give 0 where the CPU gives 9.18355e-41. The `scalar_mul_subnormal` examples differ between formats: fp16 `subnormal` passes, bf16 `subnormal` fails.
6. **`inf_nan` passes for reciprocal, `x * nan` and `x + 1e-40` in fp16, and fails elsewhere.** It fails for the other `scalar_*` examples, and for `div` the `non_finite` class fails. The paste shows no point, count or value for these failing classes.
7. **`normal` fails in `reciprocal_*`, `div_*` and the failing `scalar_*` examples**, but the paste shows no point. These classes do not involve zeros or subnormals in the input, so the cause could be accuracy beyond 1 ULP, a changed value, or a zero or non-finite result. The paste does not say which.

## The passing examples are weaker than they look

- **`scalar_add_tiny_fp16_edge` passes trivially.** `1e-40` is below the fp16 subnormal range, so it rounds to 0 in fp16 and the expected result is just x. The pass shows that adding a constant that rounds to zero changes nothing. It does not test subnormal handling. The bf16 version, where 1e-40 is representable, fails.
- **`scalar_mul_nan` passes for every class**, including `zero` (0 * NaN = NaN). It shows the NaN constant reaches the kernel and propagates, which is a smaller claim than correct handling of every non-finite constant.

## Interpretation (inferred, not proven by this run)

- **The device does not follow the IEEE rules for zero and infinity** in division, reciprocal and `0 * inf`. The pasted points show NaN in place of infinity, and 0 in place of NaN. Whether the spec requires IEEE behaviour here is a decision for the spec owner.
- **An infinity constant may not reach the kernel as infinity.** `x * inf` and `x * -inf` return 0 for x = 0, where an infinite constant would give NaN. A finite constant would give 0. This fits what the paste shows, but I have not verified it, for example by checking the stored value of the constant on the device. The same question applies to `x * 1e10` in fp16, where the constant rounds to infinity on the CPU (the fp16 maximum is 65504).
- **The lost sign of zero matches the earlier ReLU result.** In `relu/results.md` the `zero` class failed, and I had to assume the device returns +0 for -0 to reproduce it. Here the same behaviour appears for multiplication. This supports the assumption, but both cases are consistent with the same cause and the paste does not prove it.
- **The zero-handling findings are likely independent of the storage-format effect** seen in `relu` and `clamp` (nonzero values moving to a neighbouring bit pattern), since they involve zeros and infinities and not rounded values. The `normal`-class failures could be affected by that storage effect. This suite does not tell them apart.

## Context and limits

- These results come from a Spyre 1.0 card and the Spyre 1.0 software stack. They say nothing about Spyre 2.0, which is the card this assignment is preparing to test.
- The pass rule needs the result to be nonzero and within 1 ULP for ordinary results. That is stricter than the repo's `allclose` at 0.1, which is why the repo's tests did not show these differences.
- Only the first failing class of each example is printed. The counts and points of the other failing classes need a run with more detail, for example by calling the criteria directly on the device result.
- The suite covers fp16 and bf16 only, not fp32.

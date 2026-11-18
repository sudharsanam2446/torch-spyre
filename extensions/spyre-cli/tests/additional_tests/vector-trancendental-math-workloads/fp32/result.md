# fp32 suite: device results

Command: `python3 base/run_all.py fp32`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **2 of 5 passed**. Three examples failed at the compile step, so their sdk checks were skipped.

| Workload | Example | Operation | compile | sdk | detail |
|---|---|---|---|---|---|
| WK-TRN-01 | `exp_fp32` | Exponential | **FAIL** | SKIP | `torch.compile result does not match CPU` |
| WK-TRN-01 | `exp_wide_fp32` | Exponential, wide range | **FAIL** | SKIP | `torch.compile result does not match CPU` |
| WK-TRN-03 | `reciprocal_fp32` | Reciprocal | PASS | PASS | max\|diff\|=1.1920928955078125e-07 |
| WK-TRN-03 | `reciprocal_wide_fp32` | Reciprocal, wide range | PASS | PASS | max\|diff\|=7.555786372591432e+22 |
| WK-TRN-03 | `reciprocal_subnormal_fp32` | Reciprocal, subnormal | **FAIL** | SKIP | `torch.compile result does not match CPU` |

Pass criterion: `allclose` against a float64 reference rounded to fp32, with atol 1e-37 and
rtol 1e-5 (`FP32_TOL` in `fp32_examples.py`).

## Findings

1. **`exp_fp32` fails, and this was not predicted.** Its inputs are N(0, 1), so there is no
   overflow, no subnormal result and no wide range. The suite docstring expected trouble only in
   the tails of `exp_wide_fp32`. A failure on this example means the `exp` result differs from
   the reference by more than rtol 1e-5 over ordinary values.
2. **`exp_wide_fp32` fails.** Because `exp_fp32` also fails, the wide range cannot be named as
   the cause. The run does not show whether the mismatch comes from the same error as
   `exp_fp32`, from the tails (overflow near x = 88.72, subnormal results below 1.18e-38), or
   from both.
3. **`reciprocal_subnormal_fp32` fails.** This was predicted in the suite docstring: 1/x
   overflows fp32 for x < 2.94e-39, so the reference is inf for part of the sweep, and a device
   that flushes subnormal inputs to zero would return inf for all of them. This run does not
   confirm which of these happened.
4. **`reciprocal_fp32` and `reciprocal_wide_fp32` pass.** Reciprocal is accurate across
   10^-30 to 10^30. The large `max|diff|` of 7.56e22 for `reciprocal_wide_fp32` is an absolute
   difference. Outputs reach about 1e30, so it is a relative error of about 7.6e-8, well inside
   rtol 1e-5. The 1.19e-7 for `reciprocal_fp32` is 2^-23 on outputs of at most 2.
5. **No numerics for the failures.** Upstream `_compile_one` exits on mismatch, so the three
   failing examples have no sdk result. The size and location of their errors, and the
   standalone launch path, were not observed.

## Coverage

Of the fp32 gap in the "Missing Tests" sheet, the reciprocal cases (WK-TRN-03) have a
device-observed fp32 pass for normal and wide ranges. The fp32 `exp` cases (WK-TRN-01) have an
observed failure that needs a follow-up before it can be explained.

## Suggested follow-up

Measure the `exp_fp32` error instead of judging it, for example with the `ulp` suite on fp32
`exp`. That gives the size of the error and where it occurs, which pass/fail cannot.

## Limits of this run

- Only the figures above were observed. No cause is proven.
- The pass criterion is the `allclose` tolerance. It does not measure ULP error.

# rsqrt_eps suite: device results

Command: `python3 base/run_all.py rsqrt_eps`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **6 of 6 passed**. Every example built (compile PASS), launched, and passed all four input classes at the sdk step.

Operation: `rsqrt(x + 1e-6)`, the epsilon guard in the WK-TRN-02 spec, over x in [1e-7, 1e4] plus +-0 and the subnormals of each format.

| Workload | Example | Operation | compile | sdk | per-class result |
|---|---|---|---|---|---|
| WK-TRN-02 | `rsqrt_eps_scalar_fp16` | rsqrt(x + eps), scalar eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |
| WK-TRN-02 | `rsqrt_eps_scalar_bf16` | rsqrt(x + eps), scalar eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |
| WK-TRN-02 | `rsqrt_eps_scalar_fp32` | rsqrt(x + eps), scalar eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |
| WK-TRN-02 | `rsqrt_eps_tensor_fp16` | rsqrt(x + eps), tensor eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |
| WK-TRN-02 | `rsqrt_eps_tensor_bf16` | rsqrt(x + eps), tensor eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |
| WK-TRN-02 | `rsqrt_eps_tensor_fp32` | rsqrt(x + eps), tensor eps | PASS | PASS | zero=PASS subnormal=PASS small=PASS large=PASS |

## What each class checks

Every output element must be finite and within rtol of the exact value of `rsqrt(x + 1e-6)`.
rtol is 1e-4 for fp32 (the spec's value) and 2e-2 for fp16 and bf16, because the spec's 1e-4 is
below what those formats can represent.

| Class | Meaning |
|---|---|
| `zero` | x = +-0. The exact result is 1000, so the guard must be applied. Without it the result is inf. |
| `subnormal` | x is a subnormal of the format. |
| `small` | x is a normal number up to 1e-3. |
| `large` | x above 1e-3, up to 1e4. |

## What was observed

1. **The guard works in all three formats, for both forms.** `zero=PASS` means `x + 1e-6` did
   not collapse to 0 for x = +-0, so `rsqrt` returned about 1000 and not inf.
2. **fp16 is the case this suite was written for, and it passed.** In fp16 the smallest normal
   number is 6.1e-5, so 1e-6 is stored as a subnormal (1.013e-6). A device that flushed
   subnormals would turn `x + 1e-6` into `x` and `rsqrt(0)` into inf. The `zero` and
   `subnormal` classes passed, so on this run the subnormal constant and its sum with x were
   kept. This is a statement about the pass criteria (finite, within 2e-2 of exact), which the
   paste reports only as PASS. The paste has no per-element values.
3. **Scalar and tensor forms agree.** The scalar form compiles the constant as an extra kernel
   argument `(x, constant, out)` and the tensor form passes it as a second input. Both
   gave the same verdicts.
4. **No input class failed**, so there is no failing point, count or max|diff| to report.

## Context and limits

- These results come from a Spyre 1.0 card and the Spyre 1.0 software stack. They say nothing
  about Spyre 2.0, which is the card this assignment is preparing to test.
- The pass rule is a relative tolerance, and the `rsqrt_eps` suite checks only that rule.
  It does not compare bit patterns, so it cannot say whether the output is the nearest
  representable value, only that it is within 2e-2 (fp16, bf16) or 1e-4 (fp32).
- The relu and clamp suites on the same card failed their bit-exact checks, and the likely
  cause is the storage format (see `relu/results.md` and `clamp/results.md`). This suite uses
  a tolerance and not bit-exact comparison, so that storage effect would be small enough to
  pass within rtol. That is an inference from the pass criteria, not something this paste
  shows.
- For the spec's WK-TRN-02 requirement, this run shows the epsilon guard `rsqrt(x + 1e-6)`
  works on these inputs, for the six examples above (scalar and tensor forms, fp16, bf16 and fp32). It does not cover other eps
  values, other shapes, or the `ulp` accuracy of `rsqrt` (see the `ulp` suite).

# clamp suite: device results

Command: `python3 base/run_all.py clamp`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **0 of 2 passed**. Both examples built (compile PASS) and launched, and both failed the bit-exact `inside` check at the sdk step.

Operation: `torch.clamp(x, min=-1.5, max=3.5)` over every 16-bit pattern once, compared bit for bit with the CPU result.

| Workload | Example | Operation | compile | sdk | per-class result |
|---|---|---|---|---|---|
| WK-ACT-03 | `clamp_fp16_exact` | Clamp (bit-exact) | PASS | **FAIL** | inside=**FAIL** below=PASS above=PASS edges=**FAIL** special=PASS |
| WK-ACT-03 | `clamp_bf16_exact` | Clamp (bit-exact) | PASS | **FAIL** | inside=**FAIL** below=PASS above=PASS edges=PASS special=PASS |

First failure reported for each example (the first five points of the `inside` class):

`clamp_fp16_exact`: 15488 points differ, for example

| x | got | want |
|---|---|---|
| -6.10948e-05 (0x8401) | -6.11544e-05 (0x8402) | -6.10948e-05 (0x8401) |
| -6.1214e-05 (0x8403) | -6.12736e-05 (0x8404) | -6.1214e-05 (0x8403) |
| -6.13332e-05 (0x8405) | -6.13928e-05 (0x8406) | -6.13332e-05 (0x8405) |
| -6.14524e-05 (0x8407) | -6.1512e-05 (0x8408) | -6.14524e-05 (0x8407) |
| -6.15716e-05 (0x8409) | -6.16312e-05 (0x840a) | -6.15716e-05 (0x8409) |

`clamp_bf16_exact`: 24830 points differ, for example

| x | got | want |
|---|---|---|
| -9.18355e-41 (0x8001) | -0 (0x8000) | -9.18355e-41 (0x8001) |
| -1.83671e-40 (0x8002) | -0 (0x8000) | -1.83671e-40 (0x8002) |
| -2.75506e-40 (0x8003) | -0 (0x8000) | -2.75506e-40 (0x8003) |
| -3.67342e-40 (0x8004) | -0 (0x8000) | -3.67342e-40 (0x8004) |
| -4.59177e-40 (0x8005) | -0 (0x8000) | -4.59177e-40 (0x8005) |

## What was observed

1. **The clamp itself works.** `below` (saturate to -1.5), `above` (saturate to 3.5) and
   `special` (sign of zero, NaN) pass for both formats. Values outside the bounds are
   clamped to the exact bound bits.
2. **Values inside the bounds come back changed.** The `inside` class requires the output
   bits to equal the input bits, and thousands of points differ (15488 for fp16, 24830 for
   bf16). In the points shown the value was not clamped: it was altered.
   - fp16: in the five points shown, the result is one bit pattern away from the input.
     The inputs are the odd patterns 0x8401, 0x8403, ... and each came back as the next
     even one. The paste does not show whether this holds for all 15488 points.
   - bf16: tiny values, including subnormals, come back as -0.
3. **`edges` fails for fp16 only.** The edge points are the bounds and one bit pattern either
   side. The paste does not give a count. See the interpretation below.
4. **Compile PASS.** The build step in `base/check.py` compiles and runs the op once but does
   not compare numerics, so a wrong result still produces a kernel. That is why these
   examples reach the sdk step, where upstream's `bf16` and `fp32` suites would have
   stopped at the build.

## Interpretation (inferred, not proven by this run)

`torch_spyre/csrc/types_mapping.h` maps both float16 and bfloat16 to the device format
`SEN169_FP16`. I could not find a definition of that format in this repository, so what
follows is an inference from the figures above.

I tried a simple CPU model of the device and ran it through the same `clamp` criteria:

- **fp16 model:** a bit pattern with an odd lowest mantissa bit moves to the next even
  pattern in magnitude (normal values only). This gives `inside` = 15488 points and
  `edges` = FAIL, matching the paste. It also reproduces the first points above (0x8401 to
  0x8402, 0x8403 to 0x8404, and so on).
- **bf16 model:** every nonzero value with magnitude below 2^-30 (about 9.3e-10) becomes a
  signed zero. This gives `inside` = 24830 points, which is exactly all bf16 subnormals plus
  the normal values with exponent field 1 to 96, in both signs. It matches the paste, and
  `edges` passes as observed.

Both models match the reported counts and verdicts, so they are a consistent explanation. They
are not proof, because other behaviours could give the same counts. The shape of the fp16
error suggests the device format keeps one fewer mantissa bit than fp16, and the bf16 error
suggests a narrower exponent range at the small end. Check this against the format
documentation before relying on it.

If this is right, the failures come from converting values to the device format, not from the
clamp logic. The `below`, `above` and `special` classes passing supports that, and an
identity copy of the same inputs would likely show the same `inside` differences. I have not
run that comparison.

## What this means for the test

- **The bit-exact premise needs a decision.** Comparing every 16-bit pattern bit for bit
  against the CPU assumes the device stores each pattern unchanged. This run suggests it does
  not, so any value the device format cannot hold will fail `inside` whatever the kernel does.
- **Options:** keep the test as is and record the finding, or restrict `inside` to values the
  device format can represent. The second needs the format definition first.
- **Same effect elsewhere.** The `relu`, `leakyrelu` and `subnormal-zero-nonfinite-edge-cases` suites use the same
  all-patterns inputs and bit-exact classes. They may fail in the same way. No device results
  for them have been seen.

## What upstream's check would have said

With the model above, the largest absolute difference is about 0.00195 for fp16 and about
9.3e-10 for bf16, so `allclose` with atol and rtol of 0.02 would pass both. This comes from
the CPU model, not from the device. If the model is right, it is a concrete case of
`allclose` passing a result that the bit-exact check rejects.

## Limits of this run

- Only the figures in the tables above were observed. The interpretation and the last
  section come from a CPU model fitted to those figures.
- The `edges` failure count for fp16 and the failing classes' full lists were not provided.
- Only the bounds -1.5 and 3.5 were used.

# relu suite: device results

Command: `python3 base/run_all.py relu`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **0 of 2 passed**. Both examples built (compile PASS) and launched, and both failed the bit-exact check at the sdk step.

Operation: `torch.relu(x)` over every 16-bit pattern once, compared bit for bit with the CPU result.

| Workload | Example | Operation | compile | sdk | per-class result |
|---|---|---|---|---|---|
| WK-ACT-03 | `relu_fp16_exact` | ReLU (bit-exact) | PASS | **FAIL** | positive=**FAIL** negative=PASS zero=**FAIL** inf/nan=PASS |
| WK-ACT-03 | `relu_bf16_exact` | ReLU (bit-exact) | PASS | **FAIL** | positive=**FAIL** negative=PASS zero=**FAIL** inf/nan=**FAIL** |

First failure reported for each example (the first five points of the `positive` class):

`relu_fp16_exact`: 15360 points differ, for example

| x | got | want |
|---|---|---|
| 6.10948e-05 (0x0401) | 6.11544e-05 (0x0402) | 6.10948e-05 (0x0401) |
| 6.1214e-05 (0x0403) | 6.12736e-05 (0x0404) | 6.1214e-05 (0x0403) |
| 6.13332e-05 (0x0405) | 6.13928e-05 (0x0406) | 6.13332e-05 (0x0405) |
| 6.14524e-05 (0x0407) | 6.1512e-05 (0x0408) | 6.14524e-05 (0x0407) |
| 6.15716e-05 (0x0409) | 6.16312e-05 (0x040a) | 6.15716e-05 (0x0409) |

`relu_bf16_exact`: 24703 points differ, for example

| x | got | want |
|---|---|---|
| 9.18355e-41 (0x0001) | 0 (0x0000) | 9.18355e-41 (0x0001) |
| 1.83671e-40 (0x0002) | 0 (0x0000) | 1.83671e-40 (0x0002) |
| 2.75506e-40 (0x0003) | 0 (0x0000) | 2.75506e-40 (0x0003) |
| 3.67342e-40 (0x0004) | 0 (0x0000) | 3.67342e-40 (0x0004) |
| 4.59177e-40 (0x0005) | 0 (0x0000) | 4.59177e-40 (0x0005) |

## What was observed

1. **Negative inputs are handled correctly.** `negative` (finite x < 0 must give +0.0) passes
   for both formats, so the ReLU logic itself works.
2. **Positive values come back changed.** The `positive` class requires the output bits to
   equal the input bits.
   - fp16: 15360 points differ. This is exactly the number of odd positive normal bit
     patterns (0x0401, 0x0403, ... up to 0x7BFF). In the five points shown, each came back
     as the next even pattern. The paste does not show every failing point, so that this
     holds for all 15360 is an inference.
   - bf16: 24703 points differ. In the points shown, tiny positive values, including
     subnormals, came back as 0.
3. **`zero` fails for both formats.** The class requires +0.0 to stay +0.0 and -0.0 to stay
   -0.0, bit for bit. The paste does not say which zero failed or what came back.
4. **`inf/nan` fails for bf16 only.** The paste does not say whether it was +inf, -inf or
   NaN. The class requires +inf to stay +inf, -inf to become +0.0 and NaN to stay NaN.
5. **Compile PASS.** The build step in `base/check.py` compiles and runs the op once but does
   not compare numerics, so a wrong result still produces a kernel and reaches the sdk step.

## Interpretation (inferred, not proven by this run)

The same input patterns fail in the clamp suite (see `clamp/results.md`), where the clamp bounds
are applied and values inside the bounds still change. That points at how values are stored on
the device and not at the operation. `torch_spyre/csrc/types_mapping.h` maps both float16 and
bfloat16 to the device format `SEN169_FP16`. I could not find a definition of that format in
this repository, so this is an inference from the figures.

I tried a simple CPU model of the device and ran it through the same `relu` criteria. It
reproduces every pasted count and verdict:

- **fp16 model:** an odd normal pattern moves to the next even pattern in magnitude. This gives
  `positive` = 15360, `negative` and `inf/nan` passing, as observed.
- **bf16 model:** a nonzero value with magnitude below 2^-30 (about 9.3e-10) becomes a signed
  zero, and a value with magnitude of 2^32 or more, including infinity, becomes 2^32. This gives
  `positive` = 24703 and `inf/nan` = FAIL (+inf does not stay +inf), as observed. In the same
  model fp16 `inf/nan` passes, and bf16 `negative` also passes, because ReLU maps negative
  inputs to +0 regardless of the stored value.
- **`zero`:** the storage models above keep the sign of zero, and in the clamp suite the
  sign-of-zero check passed. To reproduce `zero=FAIL` I had to add the assumption that the
  device ReLU returns +0 for an input of -0. That fits the verdict but is not shown by the
  paste. CPU `torch.relu(-0.0)` keeps -0 (bits 0x8000), and a ReLU built as `max(x, 0)` may
  legitimately return +0. This is the one class that may reflect the operation itself, and it
  needs a decision on what the spec requires.

The models are a fit to the figures, so they show the explanation is consistent and not that it
is correct. A more elaborate format model I tried first did not fit.

## Are these failures valid?

- **The measurements are valid.** Both kernels compiled, launched and returned results, and the
  failing counts have a specific structure.
- **They probably do not show a ReLU defect**, apart from possibly `zero`. Every input passes
  through the device format conversion before the operation, so a plain copy would show the
  same differences.
- **A control run settles it.** Run a plain copy of the same inputs to the device and back, with
  no ReLU. From `additional_tests/base/`, on the device:

  ```python
  import torch
  from workloads_common import load_examples, make_inputs

  ex = load_examples("relu")["relu_fp16_exact"]
  (x,) = make_inputs(ex)
  y = x.to("spyre").cpu()
  ok = ~x.isnan()
  print("patterns changed by a plain copy:", (x.view(torch.int16) != y.view(torch.int16))[ok].sum().item())
  ```

  Repeat with `relu_bf16_exact`. If the copy changes about 15360 fp16 patterns, the failures
  come from the format and not from ReLU. If it changes none, the failures are in the ReLU
  kernel and the model above is wrong. This control has not been run.

## What this means for the test

- **The bit-exact premise needs a decision.** Comparing every 16-bit pattern against the CPU
  assumes the device stores each pattern unchanged. This run suggests it does not, so any value
  the device format cannot hold fails `positive` whatever the kernel does.
- **Options:** keep the test as is and record the finding, or restrict `positive` to values the
  device format can represent. The second needs the format definition first.
- **`zero` should be judged on its own.** It is the class most likely to say something about
  the operation.
- **Same effect elsewhere.** The `leakyrelu` and `subnormal-zero-nonfinite-edge-cases` suites use the same all-patterns
  inputs and bit-exact classes, and may fail in the same way. No device results for them have
  been seen.

## What upstream's check would have said

With the models above, the largest difference for fp16 is about 0.002, so `allclose` with atol
and rtol of 0.02 would pass the fp16 result. For bf16, saturating to 2^32 is a large error, so
`allclose` would reject it. This comes from the CPU models, not from the device.

## Limits of this run

- Only the figures in the tables above were observed. The interpretation and the last section
  come from a CPU model fitted to those figures.
- The paste does not show which zero failed, which infinity or NaN failed, or the full lists
  of failing points.
- No plain-copy control has been run.

# ulp suite: device results

Command: `python3 base/run_all.py ulp`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **5 examples FAIL and 19 are MEASURED** (the run summary prints "19/24 examples passed", which counts MEASURED as a pass). All 24 examples built (compile PASS) and launched.

Operation: each example runs one op over a sweep of 65536 values of the spec domain and measures the error in ULP against a float64 reference (`ulp_metric.py`). Missing Tests item 1 (no ULP measurement for any op) is the gap this suite fills.

## Why some results are MEASURED

- **PASS or FAIL** is reported only when the example has a limit. Only two ops have one: exp (max ULP error <= 1.5, WK-TRN-01 in the spec) and relu (bit-exact, so 0, WK-ACT-03).
- **MEASURED** means the example has no limit. The spec states those ops as rtol (rsqrt), atol (GELU), monotonicity and symmetry (sigmoid, tanh), a minimum location (SiLU) or matching the CPU (reciprocal, division), not as a ULP figure. Choosing a ULP limit for them would invent a threshold, so the figures are recorded and nothing is compared.
- **A MEASURED example does not mean the result is good.** It means no ULP limit was applied. The figures below are the result.
- **The `class_mismatch` count is not part of MEASURED.** It counts points where the device returns NaN, inf or a finite number in place of the other. Six MEASURED examples have a nonzero count (reciprocal fp16, bf16, fp32 and div fp16, bf16, fp32), and these read as PASS in the run output although they are real mismatches. Treat them as findings. This is how the suite was built, and I have left it unchanged.

## Results

`max_ulp`, `mean` and `p99` are over normal results. `tiny_max` is over zero and subnormal results. The limit is blank for MEASURED.

| Example | Workload | Limit | sdk | max_ulp | mean | p99 | tiny_max | class_mismatch |
|---|---|---|---|---|---|---|---|---|
| `exp_fp16_ulp` | WK-TRN-01 | 1.5 | **FAIL** | 19.245 | 1.977 | 11.704 | 8.693 | 14082 |
| `exp_bf16_ulp` | WK-TRN-01 | 1.5 | **FAIL** | 255.708 | 51.682 | 249.711 | 108.698 | 13886 |
| `exp_fp32_ulp` | WK-TRN-01 | 1.5 | **FAIL** | 15291.735 | 3993.419 | 14259.971 | 4367372.340 | 0 |
| `rsqrt_fp16_ulp` | WK-TRN-02 | | MEASURED | 2.996 | 0.799 | 2.388 | 0.000 | 0 |
| `rsqrt_bf16_ulp` | WK-TRN-02 | | MEASURED | 0.708 | 0.269 | 0.664 | 0.000 | 0 |
| `rsqrt_fp32_ulp` | WK-TRN-02 | | MEASURED | 1.625 | 0.409 | 1.223 | 0.000 | 0 |
| `reciprocal_fp16_ulp` | WK-TRN-03 | | MEASURED | 3.466 | 0.949 | 2.862 | 1.733 | **1** |
| `reciprocal_bf16_ulp` | WK-TRN-03 | | MEASURED | 254.016 | 109.259 | 250.137 | 0.000 | **24831** |
| `reciprocal_fp32_ulp` | WK-TRN-03 | | MEASURED | 9920450.926 | 10527.083 | 0.834 | 0.000 | **72** |
| `div_fp16_ulp` | WK-TRN-03 | | MEASURED | 5.572 | 0.984 | 3.134 | 2.472 | **2** |
| `div_bf16_ulp` | WK-TRN-03 | | MEASURED | 0.924 | 0.273 | 0.707 | 0.000 | **2** |
| `div_fp32_ulp` | WK-TRN-03 | | MEASURED | 1.901 | 0.387 | 1.188 | 0.000 | **2** |
| `sigmoid_fp16_ulp` | WK-TRN-04 | | MEASURED | 17.671 | 1.422 | 10.195 | 8.920 | 0 |
| `sigmoid_bf16_ulp` | WK-TRN-04 | | MEASURED | 0.928 | 0.247 | 0.811 | 0.000 | 0 |
| `tanh_fp16_ulp` | WK-TRN-04 | | MEASURED | 5.987 | 0.627 | 3.758 | 0.000 | 0 |
| `tanh_bf16_ulp` | WK-TRN-04 | | MEASURED | 1.076 | 0.114 | 0.782 | 0.000 | 0 |
| `gelu_fp16_ulp` | WK-ACT-01 | | MEASURED | 2047.440 | 70.533 | 1694.293 | 1023.488 | 0 |
| `gelu_bf16_ulp` | WK-ACT-01 | | MEASURED | 255.195 | 37.274 | 247.686 | 0.000 | 0 |
| `gelu_tanh_fp16_ulp` | WK-ACT-01 | | MEASURED | 2047.138 | 63.295 | 1660.605 | 1014.418 | 0 |
| `gelu_tanh_bf16_ulp` | WK-ACT-01 | | MEASURED | 255.695 | 37.702 | 247.619 | 0.000 | 0 |
| `silu_fp16_ulp` | WK-ACT-02 | | MEASURED | 16.457 | 1.734 | 10.697 | 0.000 | 0 |
| `silu_bf16_ulp` | WK-ACT-02 | | MEASURED | 1.136 | 0.291 | 0.849 | 0.000 | 0 |
| `relu_fp16_ulp` | WK-ACT-03 | 0 | **FAIL** | 1.000 | 0.500 | 1.000 | 0.000 | 0 |
| `relu_bf16_ulp` | WK-ACT-03 | 0 | **FAIL** | 255.000 | 143.748 | 254.000 | 127.000 | 0 |

## Worst point of each example

| Example | Worst point (x, got, want, ulp) |
|---|---|
| `exp_fp16_ulp` | x=-8.36719, 0.000230074, 0.000232339, 19.245 |
| `exp_bf16_ulp` | x=-22.875, 0, 1.16415e-10, 255.708 |
| `exp_fp32_ulp` | x=15.9424, 8.38091e+06, 8.38856e+06, 15291.735 |
| `rsqrt_fp16_ulp` | x=7.09891e-05, 118.5, 118.688, 2.996 |
| `rsqrt_bf16_ulp` | x=2.03028e-07, 2208, 2224, 0.708 |
| `rsqrt_fp32_ulp` | x=0.00109868, 30.1692, 30.1692, 1.625 |
| `reciprocal_fp16_ulp` | x=-8536, -0.000116944, -0.000117123, 3.466 |
| `reciprocal_bf16_ulp` | x=-6.38777e+29, -0, -1.5654e-30, 254.016 |
| `reciprocal_fp32_ulp` | x=-2.96679e-39, -1.35854e+38, -3.37064e+38, 9920450.926 |
| `div_fp16_ulp` | a=-41.0938, b=21.3125, -1.93359, -1.92773, 5.572 |
| `div_bf16_ulp` | a=0.00180817, b=0.00726318, 0.248047, 0.249023, 0.924 |
| `div_fp32_ulp` | a=-63.2968, b=-0.0039056, 16206.7, 16206.7, 1.901 |
| `sigmoid_fp16_ulp` | x=-9.07031, 0.000113964, 0.000115037, 17.671 |
| `sigmoid_bf16_ulp` | x=-1.96875, 0.12207, 0.122559, 0.928 |
| `tanh_fp16_ulp` | x=-0.250732, -0.246338, -0.245605, 5.987 |
| `tanh_bf16_ulp` | x=-0.178711, -0.175781, -0.176758, 1.076 |
| `gelu_fp16_ulp` | x=-3.64453, 0, -0.000488043, 2047.440 |
| `gelu_bf16_ulp` | x=-4.34375, 0, -3.03984e-05, 255.195 |
| `gelu_tanh_fp16_ulp` | x=-3.55859, 0, -0.000488043, 2047.138 |
| `gelu_tanh_bf16_ulp` | x=-6.1875, 0, -1.45519e-11, 255.695 |
| `silu_fp16_ulp` | x=-9.92969, -0.000479698, -0.000483513, 16.457 |
| `silu_bf16_ulp` | x=0.00726318, 0.00366211, 0.00364685, 1.136 |
| `relu_fp16_ulp` | x=6.10948e-05, 6.11544e-05, 6.10948e-05, 1.000 |
| `relu_bf16_ulp` | x=2.34181e-38, 0, 2.34181e-38, 255.000 |

The first class-mismatch point reported for each example with a nonzero count:

| Example | Point (input, got, want) |
|---|---|
| `exp_fp16_ulp` | x=11.0859, inf, 65248 |
| `exp_bf16_ulp` | x=22.875, nan, 8.58993e+09 |
| `reciprocal_fp16_ulp` | x=0, nan, inf |
| `reciprocal_bf16_ulp` | x=-9.27685e-10, nan, -1.08213e+09 |
| `reciprocal_fp32_ulp` | x=-2.88551e-39, -1.36608e+38, -inf |
| `div_fp16_ulp` | a=13.6641, b=0, nan, inf |
| `div_bf16_ulp` | a=13.6875, b=0, nan, inf |
| `div_fp32_ulp` | a=13.6649, b=0, 6.14299e-19, inf |

## What was observed

1. **exp fails its 1.5 ULP limit in all three formats**, by a wide margin: 19.2 ULP in fp16, 255.7 in bf16, 15291.7 in fp32. For fp32 the worst point is about 0.09% away from the exact value, computed from the pasted figures (8.38091e+06 against 8.38856e+06). The class mismatches in fp16 and bf16 are overflow cases: at x=11.0859 the device returns inf, where the exact fp16 result is 65248 and fits in fp16 (its maximum is 65504), and at x=22.875 bf16 returns NaN.
2. **relu fails the bit-exact limit.** fp16 is 1 ULP off at x=6.10948e-05, where an odd bit pattern came back as the next pattern. This is the same first point as in `relu/results.md`. bf16 is 255 ULP off, because x=2.34181e-38 came back as 0.
3. **Division and reciprocal by zero** return NaN in fp16 and bf16, where the exact value is inf, as in the edge-cases results. In fp32 `div` by 0 returns 6.14299e-19, which is neither inf nor NaN.
4. **reciprocal fp32 is accurate for ordinary values** (p99 0.834 ULP), but the worst point is for a subnormal input, x=-2.96679e-39, where the device returns -1.36e+38 and the exact value is -3.37e+38. The first of the 72 class mismatches is x=-2.88551e-39, where the exact result overflows to -inf and the device returns a finite number. The paste shows only that first mismatch, so the range and kind of the other 71 are not known.
5. **GELU returns 0 for small negative outputs.** All four GELU worst points have got=0 where the exact value is small but nonzero, for example -0.000488 at x=-3.64 in fp16. This gives 2047 ULP in fp16 and 255 in bf16. The paste does not show where the device starts returning 0.
6. **rsqrt, sigmoid, tanh and SiLU are within a few ULP in bf16** (0.71 to 1.14). In fp16 the same ops are further out: rsqrt 3.0, tanh 6.0, SiLU 16.5, sigmoid 17.7. The worst fp16 points for sigmoid and SiLU are at large negative x (-9.07 and -9.93), where the outputs are about 1e-4.
7. **No figure here can be compared with a spec limit**, apart from exp and relu, because the other ops have none in ULP.

## Interpretation (inferred, not proven by this run)

- **The relu and bf16 reciprocal figures match the storage effect in `relu/results.md` and `clamp/results.md`.** On the CPU, the reciprocal bf16 sweep has 24830 nonzero values below 2^-30 in magnitude, plus one zero. That is 24831, the reported `class_mismatch`. A nonzero input below 2^-30 appears to become zero on the device, and 1/0 then returns NaN, as in the pasted point at x=-9.27685e-10. The worst point, x=-6.38777e+29, has an exact result of -1.5654e-30, which is also below 2^-30 and came back as -0. This fits but I have not tested it.
- **The fp16 odd-pattern change is consistent with relu fp16.** The relu fp16 worst point is the same pattern that `relu/results.md` showed (0x0401 came back as 0x0402). That example sweeps every 16-bit pattern, so a plain copy would likely show the same difference.
- **Part of the ULP error may therefore come from the stored format, not from the operation.** For example, a 255 ULP bf16 error for a value near 1e-38 is far larger than any rounding in the op. A control run, copying the same inputs to the device and back with no op, would separate the two. I have not run it.

## Context and limits

- These results come from a Spyre 1.0 card and the Spyre 1.0 software stack. They say nothing about Spyre 2.0, which is the card this assignment is preparing to test.
- Only the worst point and the first class mismatch of each example are printed. The location of the other mismatches, and the distribution of the error across the domain, are not in the paste.
- 65536 values are exhaustive in fp16 and bf16 (every value of the sweep domain), and a sample in fp32.
- The suite has no INT8, no LeakyReLU or clamp ULP example, and no FP32 for tanh, sigmoid, SiLU and GELU. See the "Missing Tests" sheet for items 4 to 6.

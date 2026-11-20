# sigmoid_tanh suite: device results

Command: `python3 base/run_all.py sigmoid_tanh`, run from `additional_tests/` on a device with a physical accelerator card.

Result: **2 of 4 passed**. All four examples built (compile PASS) and launched. Both bf16 examples passed. Both fp16 examples failed at the sdk step.

Operation: `sigmoid` and `tanh` over every 16-bit pattern once, which includes every representable value of [-16, 16]. The properties are checked on the finite inputs with |x| <= 16.

| Workload | Example | Operation | compile | sdk | per-class result |
|---|---|---|---|---|---|
| WK-TRN-04 | `sigmoid_fp16_props` | Sigmoid | PASS | **FAIL** | monotonic=**FAIL** bounded=PASS symmetry=**FAIL** anchors=**FAIL** |
| WK-TRN-04 | `sigmoid_bf16_props` | Sigmoid | PASS | PASS | monotonic=PASS bounded=PASS symmetry=PASS anchors=PASS |
| WK-TRN-04 | `tanh_fp16_props` | Tanh | PASS | **FAIL** | monotonic=PASS bounded=PASS symmetry=PASS anchors=**FAIL** |
| WK-TRN-04 | `tanh_bf16_props` | Tanh | PASS | PASS | monotonic=PASS bounded=PASS symmetry=PASS anchors=PASS |

## What each class checks

| Class | Meaning |
|---|---|
| `monotonic` | Sorted by x, the output never goes down (non-strict, since both functions round to exactly 1.0 well inside the domain). |
| `bounded` | sigmoid in [0, 1], tanh in [-1, 1], no NaN. |
| `symmetry` | tanh(-x) = -tanh(x) within 1 ULP. sigmoid(x) + sigmoid(-x) = 1 within 1 ULP of 1. |
| `anchors` | At x = 0, +-1, +-4 and +-16 the result is within 1 ULP of the CPU result. |

## First failure reported for each failing class

The run prints only the first failing class of each example, so the details below are all the
paste contains.

`sigmoid_fp16_props`, `monotonic`: 2 steps go down. Both are listed:

| x | got | next x | got at next x |
|---|---|---|---|
| -1.81934 (0xbf47) | 0.139648 | -1.81836 | 0.139404 |
| -0.606934 (0xb8db) | 0.353027 | -0.606445 | 0.352539 |

In both pairs the output falls by 2 ULP of fp16 (0.000244 in [0.125, 0.25) and 0.000488 in
[0.25, 0.5)), where a non-decreasing function must stay level or rise.

`tanh_fp16_props`, `anchors`: 2 points differ, both anchors at x = +-1:

| x | got | want (CPU) |
|---|---|---|
| -1 (0xbc00) | -0.762695 (0xba1a) | -0.761719 (0xba18) |
| 1 (0x3c00) | 0.762695 (0x3a1a) | 0.761719 (0x3a18) |

The device result is 2 ULP away from the CPU result (bit patterns 0x3a1a and 0x3a18), and the
allowed distance is 1 ULP. The absolute difference is about 0.001.

## What was observed

1. **fp16 and bf16 give different verdicts for the same operation.** Both bf16 examples pass
   all four classes. Both fp16 examples fail.
2. **Sigmoid fp16 is not monotonic.** Two adjacent pairs of inputs are in the wrong order in
   the output. The paste lists both, and the count is 2.
3. **Sigmoid fp16 also fails `symmetry` and `anchors`.** The paste does not show the details,
   because only the first failing class is printed. I do not know the count of failing points,
   which anchors differ, or how far off they are.
4. **Tanh fp16 fails only at the +-1 anchors.** Its monotonic, bounded and symmetry classes
   pass, and the other anchors (0, +-4, +-16) are within 1 ULP. Because tanh `symmetry` passes,
   the +1 and -1 results are the same distance off in opposite directions, as the table shows.
5. **`bounded` passes everywhere.** No output left [0, 1] for sigmoid or [-1, 1] for tanh.

## Interpretation (inferred, not proven by this run)

- **The errors shown are small in absolute terms.** The fp16 differences in the pasted points
  are about 1e-3 or less (0.000244, 0.000488, about 0.001). A tolerance check such as
  `allclose` at atol = rtol = 0.1 would very likely not have caught them. They fail here
  because this suite checks the order of neighbouring outputs and a 1 ULP limit. The sizes for
  the sigmoid `symmetry` and `anchors` failures are not in the paste, so this holds only for
  the points listed.
- **Why bf16 passes while fp16 fails is not known.** One possibility is that the device
  computes both formats with the same internal accuracy, and that error is 2 ULP in fp16 but
  well under 1 ULP in bf16, because a bf16 ULP is 8 times larger (2^-7 against 2^-10 relative).
  That fits the figures, but I have not tested it, for example by comparing the same inputs in
  another format.
- **The tanh result is likely a ULP-level accuracy finding.** At x = 1 the device returns
  0.762695. The CPU fp16 result 0.761719 is the nearest fp16 value to the true tanh(1)
  (0.761594), so the device result is 2 ULP high. Whether Spyre accepts a 2 ULP error for fp16
  tanh is a spec decision. The `ulp` suite measures this directly and has not been run yet.
- **The storage-format effect seen in relu and clamp does not obviously explain this.** In
  `relu/results.md` and `clamp/results.md`, fp16 odd bit patterns appeared to move to the next
  even pattern. The first failing sigmoid point starts at an odd pattern (0xbf47), but that
  model would move it to a more negative input, which should lower the output, and the paste
  shows it higher than its neighbour's. I have not claimed that model explains this result.

## Context and limits

- These results come from a Spyre 1.0 card and the Spyre 1.0 software stack. They say nothing
  about Spyre 2.0, which is the card this assignment is preparing to test.
- The property rules use exhaustive input, so a pass is meaningful: bf16 sigmoid and tanh met all
  four properties on every representable value in [-16, 16].
- A passing bf16 result does not cover fp16, and the fp16 fail does not cover bf16. The
  verdicts differ and should be reported per format.
- Next step if the tanh and sigmoid fp16 numbers need explaining: run the `ulp` suite on fp16
  `tanh` and `sigmoid`, which reports the maximum error in ULP over the whole domain.

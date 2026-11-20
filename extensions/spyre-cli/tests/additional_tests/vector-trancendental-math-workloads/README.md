# spyre-cli additional tests: Vector & Transcendental Math workloads

Extra spyre-cli kernels for **WK-TRN-01..04** and **WK-ACT-01..03**, covering the
gaps listed in the "Missing Tests" sheet of
`extensions/vector-trancendental-math-workloads/Vector and Trancendental Math Workloads Base.xlsx`.
Nothing under `examples/`, `spyre_cli/`, `tests/test_examples.py` or
`tests/test_core.py` is modified, and nothing is copied from it.

```
tests/additional_tests/
  base/
    workloads_common.py     Example type, ops, inputs, CPU reference, loader
    sweeps.py               input sweeps (every 16-bit pattern, ULP, eps, div grid)
    criteria.py             pass criteria (bit-exact, monotonic, eps guard, ...)
    check.py                customised build + SDK check (see below)
    run_all.py              runs suites one by one, prints a table
  vector-trancendental-math-workloads/
    test_workloads.py       pytest wrapper + CPU tests of every criterion
    bf16/ fp32/             allclose suites, checked by upstream code
    ulp/ sigmoid_tanh/ relu/ clamp/ leakyrelu/ rsqrt_eps/ subnormal-zero-nonfinite-edge-cases/
                            suites checked by base/check.py
    ulp/ulp_metric.py       the ULP metric and its float64 references (ulp only)
```

A suite is only a registry, `<suite>/<suite>_examples.py` with
`EXAMPLES = {name: Example}`.

## Two ways an example is built and checked

`run_all.py` runs each example in two fresh processes (build, then check), because
a device fault leaves the stream in an error state for the rest of the process.
Which code runs depends on `Example.check`:

| `Example.check` | Suites | Build | Check |
|---|---|---|---|
| `""` | `bf16`, `fp32` | upstream `examples.py` `_compile_one` | upstream `check.py` `check_sdk`: `allclose` against the CPU reference |
| a criteria name, or `"ulp"` | the other seven | `base/check.py` `compile_one` | `base/check.py` `check_sdk`: the criteria of `base/criteria.py`, or the ULP measurement |

**Why a customised `check.py` for the second group.** Upstream's `_compile_one`
compares the compiled result with `torch.allclose`, which rejects a NaN, and
these sweeps hold NaN, inf, zeros and subnormals on purpose. More importantly,
one `allclose` cannot express what the sheet asks for: bit-exact output, the sign
of zero, monotonicity, symmetry, an error in ULP, a guard that must keep a result
finite. `base/check.py` therefore builds without comparing numerics (a kernel
whose results are wrong is exactly the one worth measuring) and judges the result
with a set of criteria. It prints the same `<name> sdk: <status> <detail>` line
`run_all.py` parses, so the table is the same.

`check_sdk` also reports an exception raised by `spyre_cli.launch` on the result
line, and passes the extra 0-d constant argument that the scalar eps guard
`rsqrt(x + 1e-6)` takes (`(x, constant, out)`), which upstream's check cannot do.

Status is `PASS` or `FAIL`; `MEASURED` for a ULP example without a limit, which
counts as passing because there is nothing to pass.

## Run

On a machine with a Spyre card, from `tests/additional_tests/`:

```bash
python3 base/run_all.py                          # every suite, one by one
python3 base/run_all.py bf16 fp32                # allclose suites
python3 base/run_all.py ulp                      # one suite
python3 base/run_all.py relu clamp leakyrelu     # in the order given
python3 base/run_all.py exp_fp16_ulp tanh_bf16_props   # or just some examples
python3 base/run_all.py --out /tmp/kernels       # build elsewhere (<out>/<suite>/<name>)
SPYRE_MAX_ULP=1.5 python3 base/run_all.py ulp    # gate every ULP example at 1.5

python3 -m pytest vector-trancendental-math-workloads/test_workloads.py -v
python3 -m pytest vector-trancendental-math-workloads/test_workloads.py -v -k subnormal-zero-nonfinite-edge-cases
```

A failing example does not stop the run; if it fails to build its check is
`SKIP`. The exit status is 0 only if every selected example builds and passes.
`--group` selects another folder under `additional_tests/`.

## Gaps covered

| Sheet item | Suite | Examples | Verdict |
|---|---|---|---|
| 1 no ULP measurement | `ulp` | 24: exp, rsqrt, reciprocal, div (fp16, bf16, fp32); sigmoid, tanh, gelu, gelu_tanh, silu, relu (fp16, bf16) | exp <= 1.5 ULP, relu 0 ULP, the rest `MEASURED` |
| 3 no BF16 | `bf16` | 11 | `allclose`, atol = rtol = 0.02 |
| 4 no FP32 | `fp32` | 5 | `allclose`, (1e-37, 1e-5) |
| 5 no LeakyReLU | `leakyrelu` | 4: fp16 / bf16, slopes 0.01 and 0.2 | positive bit-exact, negative within 1 ULP, subnormals, zeros, inf/nan |
| 7 no bit-exact | `relu`, `clamp` | 2 + 2 | every 16-bit pattern, compared bit for bit |
| 8 no monotonicity / symmetry | `sigmoid_tanh` | 4 | monotonic, bounded, symmetry, anchors |
| 9 subnormal / zero-division weak | `subnormal-zero-nonfinite-edge-cases` | 14 `scalar_*` | NaN kept, inf / zero bit-exact, others within 1 ULP |
| 10 reciprocal hides near-zero | `subnormal-zero-nonfinite-edge-cases` | 4: reciprocal and div, fp16 / bf16 | same |
| 12 eps guard 1e-6 | `rsqrt_eps` | 6: scalar / tensor form, fp16 / bf16 / fp32 | result finite and within rtol of `rsqrt(x + 1e-6)` |

Not covered: 2 (domain sweep as such; the `ulp`, `bf16` and `fp32` suites do sweep
the spec domains), 6 (INT8: there is no pointwise INT8 op), 11 (padded-lane
safety: every example uses 65536 elements, a multiple of the stick and of 32
cores), 13 (exact vs tanh GELU as a separate comparison), 14 (SiLU minimum near
x = -1.28), 15 (the six CLI examples; `spyre launch` is not run here), 16
(cross-core determinism).

## Criteria

Defined in `base/criteria.py`; each returns `None` or a message naming the first
offending points. The result line lists `label=PASS|FAIL` for each, then the first
failure.

| `check` | Labels | Rule |
|---|---|---|
| `relu` | positive, negative, zero, inf/nan | bits equal the CPU's; +0.0 for negatives; zeros keep their sign |
| `clamp` | inside, below, above, edges, special | bits equal the CPU's; `edges` = each bound and one step either side |
| `leakyrelu` | positive, neg, subnorm, zero, inf/nan | negative branch within 1 ULP of `x * slope` (slope rounded to the dtype); a device that flushes subnormals fails only `subnorm` |
| `sigtanh` | monotonic, bounded, symmetry, anchors | over finite \|x\| <= 16: non-decreasing; sigmoid in [0, 1], tanh in [-1, 1]; tanh(-x) = -tanh(x) and sigmoid(x) + sigmoid(-x) = 1 within 1 ULP; anchors at 0, +-1, +-4, +-16 within 1 ULP of the CPU (a constant output would otherwise pass the other three) |
| `eps` | zero, subnormal, small, large | finite and within rtol (fp32 1e-4; fp16 / bf16 2e-2: the spec's 1e-4 is below what 16 bits can represent) of `rsqrt(x + 1e-6)` |
| `edge` | zero, inf_nan, subnormal, normal (div: zero_divisor, non_finite, tiny_divisor, tiny_numerator, normal) | NaN stays NaN; inf and zero match bit for bit; other results are finite, nonzero and within 1 ULP. A scalar example also accepts the result with the scalar first rounded to the dtype |
| `ulp` | | `ulp_error = abs(got - ref) / spacing(ref)` against a float64 reference, split into normal / tiny / special. With a limit: FAIL above it, or on any NaN / inf class mismatch. Without a limit the figures are only recorded (`MEASURED`) |

## Expected findings (predictions, not observed results)

`torch_spyre/csrc/types_mapping.h` maps bfloat16 to the device FP16 format
(SEN169_FP16). From that and the code:

- `exp_wide_bf16` is likely to fail (confirmed on device, see `bf16/results.md`).
  `rsqrt_bf16` was also predicted to fail but passed. bf16 examples whose values
  lie outside the FP16 range may show large ULP errors.
- Read the bf16 `gelu` and `gelu_tanh` ULP figures with this in mind: torch's own
  bf16 CPU result already shows about 255 ULP in the far negative tail, because
  `0.5 * x * (1 + erf(x / sqrt 2))` cancels to -0 there, while the float64
  reference stays accurate. The figure measures the kernel against the true
  value, so a device matching torch's bf16 formula would show the same.
- LeakyReLU has no lowering; it may compile through the default decomposition or
  fail at build. Either is the finding.
- `rsqrt_eps_*_fp16`: 1e-6 is an fp16 subnormal. A device that flushes subnormals
  returns inf at x = 0, which `zero` reports.
- Scalar constants (`scalar_*`, `rsqrt_eps_scalar_*`, `leaky_relu_*`) may compile as
  an extra kernel argument. `run_all.py` reports a launch that raises.
- `lower_gelu` ignores `approximate`, so the `gelu_tanh` examples may differ from
  the tanh reference in the tails.

## Limitations

- **Validated on CPU only.** The imports, registries, inputs, every criterion
  (it accepts the CPU result and rejects each fault it exists for), `check_sdk`
  against a stand-in for `spyre_cli.launch`, and `run_all.py` with no device (it
  fails at the compile step, the check is `SKIP`, exit 1). Nothing has run on a
  Spyre card, and no device result is claimed.
- `spyre launch` (the CLI check in upstream's `check.py`) is not run; only the SDK
  path is.
- Examples must compile to a single kernel.
- The ULP figures are measured against float64 references. For gelu the
  references use `erfc` and a rearranged tanh form because torch's own float64
  `gelu` cancels to -0.0 below about x = -6.
- A ULP example without a limit is always `MEASURED`, even if it has NaN / inf
  class mismatches (they are counted in the line); a limit, from the example or
  from `SPYRE_MAX_ULP`, turns them into a `FAIL`.

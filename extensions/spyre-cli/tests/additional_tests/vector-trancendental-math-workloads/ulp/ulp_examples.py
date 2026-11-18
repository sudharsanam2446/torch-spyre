# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""ULP-error examples ("Missing Tests" item 1: no ULP measurement for any op).

The spec's main accuracy metric is ULP, while every tolerance in the repo is
0.1, 0.01 or 1e-3. Each example compiles one op in one precision, runs it over a
dense sweep of its spec domain (65536 values; exhaustive in fp16 and bf16) and
measures the error in ULP against a float64 reference (``ulp_metric.py`` in this folder).

Names are ``<op>_<precision>_ulp``:

    ======================  ===========  ==============  ===================
    op                      workload     precisions      domain swept
    ======================  ===========  ==============  ===================
    exp                     WK-TRN-01    fp16 bf16 fp32  [-88, 88]
    rsqrt                   WK-TRN-02    fp16 bf16 fp32  [1e-7, 1e4]
    reciprocal              WK-TRN-03    fp16 bf16 fp32  all finite values
    div                     WK-TRN-03    fp16 bf16 fp32  [-64, 64] pairs
    sigmoid, tanh           WK-TRN-04    fp16 bf16       [-16, 16]
    gelu, gelu_tanh         WK-ACT-01    fp16 bf16       [-10, 10]
    silu                    WK-ACT-02    fp16 bf16       [-10, 10]
    relu                    WK-ACT-03    fp16 bf16       [-1e4, 1e4]
    ======================  ===========  ==============  ===================

Not covered: FP32 tanh / SiLU / GELU (not in ``SPYRE_FP32_OPS``), INT8 (no
pointwise INT8 op).

Limits. Only two ops have an ULP limit in the spec or follow from it: exp (Max
ULP error <= 1.5, WK-TRN-01) and relu (bit-exact, so 0, WK-ACT-03). All others
are MEASURED: the figures are recorded but there is no pass/fail, since the spec
states them as rtol / atol / monotonicity instead. Set ``SPYRE_MAX_ULP=X`` to
gate every example at X.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example, symmetric  # noqa: E402

ALL = ("fp16", "bf16", "fp32")
HALF = ("fp16", "bf16")  # no FP32 kernel for these (see the module docstring)

# (op, workload, title, sweep spec, precisions, ULP limit or None to only measure)
_TABLE = [
    # exp overflows fp16 above x = 11.09: those inputs must give inf, and the
    # class check reports it if they do not.
    ("exp", "WK-TRN-01", "Exponential", symmetric(88), ALL, 1.5),
    # Log-uniform over the whole spec domain, one binade at a time.
    ("rsqrt", "WK-TRN-02", "Reciprocal square root", "bits:1e-7:1e4", ALL, None),
    # Every magnitude, down to subnormals: 1/subnormal overflows to inf.
    ("reciprocal", "WK-TRN-03", "Reciprocal", "bits:-1e30:1e30", ALL, None),
    # Numerator and denominator are independent permutations of the sweep.
    ("div", "WK-TRN-03", "Division", symmetric(64), ALL, None),
    ("sigmoid", "WK-TRN-04", "Sigmoid", symmetric(16), HALF, None),
    ("tanh", "WK-TRN-04", "Tanh", symmetric(16), HALF, None),
    ("gelu", "WK-ACT-01", "GELU (exact)", symmetric(10), HALF, None),
    ("gelu_tanh", "WK-ACT-01", "GELU tanh approximation", symmetric(10), HALF, None),
    ("silu", "WK-ACT-02", "SiLU", symmetric(10), HALF, None),
    # Full finite range, exhaustive in fp16 and bf16. Bit-exact: limit 0.
    ("relu", "WK-ACT-03", "ReLU", "bits:-inf:inf", HALF, 0.0),
]


def _example(op, workload, title, spec, fmt, limit):
    shapes = (SHAPE16,) * (2 if op == "div" else 1)
    return Example(
        op, workload, title, shapes, SHAPE16, EXACT, fmt,
        init=f"sweep={spec}", params={"limit": limit}, check="ulp",
    )  # fmt: skip


EXAMPLES = {
    f"{op}_{fmt}_ulp": _example(op, workload, title, spec, fmt, limit)
    for op, workload, title, spec, fmts, limit in _TABLE
    for fmt in fmts
}

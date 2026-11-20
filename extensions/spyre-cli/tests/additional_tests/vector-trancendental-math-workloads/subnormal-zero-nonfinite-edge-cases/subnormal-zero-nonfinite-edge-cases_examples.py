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


"""Subnormal, zero and non-finite edge cases (WK-TRN-03, "Missing Tests" 9 and 10).

- Item 9, "Subnormal and zero-division tests are weak or disabled": the
  ``scalar_*`` examples run the operations of the loose, ``xfail`` or skipped
  tests in ``test_inductor_dtype_scalars.py`` (``x * 6e-8``, ``x / 0.0``,
  ``x * nan``, ``x * inf``, ``x * -inf``, ``x * 1e10``, ``x + 1e-40``) on every
  16-bit pattern.
- Item 10, "Reciprocal test hides the problem": ``test_unary_op`` replaces every
  |x| < FP16_EPS by FP16_EPS before calling reciprocal. ``reciprocal_*`` and
  ``div_*`` run exactly the zeros and subnormals it removes.

Names are ``<fn>_<fmt>_edge``, fp16 and bf16. reciprocal and every scalar example
get every 16-bit pattern. div gets a 256 x 256 grid of numerator x denominator
patterns holding both zeros, both infinities, NaN, the smallest and largest
subnormals, the smallest normal, +-1..3, the largest finite values, many more
subnormals on the denominator axis, and evenly spaced finite values.

A scalar example has two CPU references and a device element passes if it
matches either: PyTorch's own ``x * 6e-8`` (the scalar kept in float32), and the
scalar first rounded to the tensor dtype, which is what a kernel holding its
constant in 16 bits computes. Which one the card follows is not assumed.

Pass rule (``edge``): NaN stays NaN; an infinity or zero matches bit for bit;
any other result is finite, nonzero and within 1 ULP of the CPU. Criteria are
split by class of input (zero / inf_nan / subnormal / normal for reciprocal and
the scalar ops; zero_divisor / non_finite / tiny_divisor / tiny_numerator /
normal for div).

Compiling a scalar example adds the constant as an extra kernel argument, so the
bundle takes ``(x, constant, out)``. ``base/check.py`` (``launch_args``) passes
it, built the way the compiled graph builds it (``spyre_constant_tensor``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

# key -> (binary op, Python scalar), from tests/inductor/test_inductor_dtype_scalars.py
SCALARS = {
    "mul_subnormal": ("mul", 6e-8),  # test_subnormal_scalar_operations
    "div_by_zero": ("div", 0.0),  # test_division_by_zero_result
    "mul_nan": ("mul", float("nan")),  # test_non_finite_scalar_inputs
    "mul_inf": ("mul", float("inf")),
    "mul_neg_inf": ("mul", float("-inf")),
    "mul_overflow": ("mul", 1e10),  # test_fp16_overflow_underflow
    "add_tiny": ("add", 1e-40),
}


def _core(fn, fmt):
    two = fn == "div"
    return Example(
        fn, "WK-TRN-03", {"reciprocal": "Reciprocal", "div": "Division"}[fn],
        (SHAPE16,) * (2 if two else 1), SHAPE16, EXACT, fmt,
        init="grid" if two else "bits16", check="edge",
    )  # fmt: skip


def _scalar(key, fmt):
    kind, scalar = SCALARS[key]
    return Example(
        "scalar_binary", "WK-TRN-03", f"x {kind} {scalar:g}", (SHAPE16,), SHAPE16,
        EXACT, fmt, init="bits16", params={"kind": kind, "scalar": scalar},
        check="edge",
    )  # fmt: skip


EXAMPLES = {
    **{f"{fn}_{fmt}_edge": _core(fn, fmt) for fn in ("reciprocal", "div") for fmt in ("fp16", "bf16")},
    **{f"scalar_{key}_{fmt}_edge": _scalar(key, fmt) for key in SCALARS for fmt in ("fp16", "bf16")},
}

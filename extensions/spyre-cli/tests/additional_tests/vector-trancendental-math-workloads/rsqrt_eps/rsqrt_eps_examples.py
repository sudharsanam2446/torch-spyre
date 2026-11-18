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


"""The epsilon guard ``rsqrt(x + 1e-6)`` (WK-TRN-02, "Missing Tests" item 12).

The spec uses eps = 1e-6; the repo tests add ``FP16_EPS`` (9.8e-4) or ``FP32_EPS``
(1.2e-7) to ``abs(randn)``, which are other numbers and never come near zero.
These examples run the spec's guard itself over x in [1e-7, 1e4] plus +-0 and the
subnormals of each format.

In fp16 the smallest normal number is 6.1e-5, so 1e-6 is stored as a *subnormal*
(1.013e-6, 1.3% off). The guard works only if the device keeps that subnormal
constant and its sum with x. A device that flushes subnormals to zero turns
``x + 1e-6`` into ``x``, and ``rsqrt(0)`` is inf, which is what the guard exists
to prevent. In bf16 and fp32 1e-6 is a normal number.

Six examples, ``rsqrt_eps_<form>_<fmt>``:

- ``form`` is ``scalar``, ``rsqrt(x + 1e-6)`` with the constant written in the
  function as the spec writes it, or ``tensor``, ``rsqrt(x + e)`` with ``e`` a
  second input filled with 1e-6. The scalar form compiles the constant as an
  extra kernel argument, ``(x, constant, out)``, which upstream ``check.py``
  cannot supply; ``base/check.py`` passes it (``launch_args``).
- ``fmt`` is fp16, bf16 or fp32.

Criteria (``eps``), by class of x: zero (result 1000), subnormal, small (normal,
<= 1e-3) and large; each result must be finite and within rtol of the exact value
(fp32 1e-4, the spec's; fp16 / bf16 2e-2, since the spec's 1e-4 is below what
those formats can represent).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

EPS = 1e-6  # the spec's guard

EXAMPLES = {
    f"rsqrt_eps_{form}_{fmt}": Example(
        "rsqrt_eps", "WK-TRN-02", f"rsqrt(x + eps), {form} eps",
        (SHAPE16,) * (2 if form == "tensor" else 1), SHAPE16, EXACT, fmt,
        init="eps", params={"form": form, "eps": EPS}, check="eps",
    )  # fmt: skip
    for form in ("scalar", "tensor")
    for fmt in ("fp16", "bf16", "fp32")
}

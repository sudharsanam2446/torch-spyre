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


"""Sigmoid / tanh monotonicity and symmetry (WK-TRN-04, "Missing Tests" item 8).

The spec asks for *monotonicity preservation* and a *symmetry check* on sigmoid
and tanh over x in [-16, 16], in FP16 and BF16. A tolerance test cannot show
either: a piecewise approximation with a kink of a few ULP, or a one-sided one,
passes ``atol = rtol = 0.1`` on random inputs.

Four examples, ``sigmoid_<fmt>_props`` and ``tanh_<fmt>_props``. The input holds
every 16-bit pattern once, which includes every representable value of
[-16, 16], so the sweep is exhaustive rather than a sampled grid. Criteria
(``base/criteria.py``, ``sigtanh``), over the finite inputs with |x| <= 16:

- monotonic: sorted by x, the output never goes down (non-strict; both functions
  round to exactly 1.0 well inside the domain);
- bounded: sigmoid in [0, 1], tanh in [-1, 1], no NaN;
- symmetry: tanh(-x) = -tanh(x) within 1 ULP; sigmoid(x) + sigmoid(-x) = 1 within
  1 ULP of 1;
- anchors: at x = 0, +-1, +-4 and +-16 the result is within 1 ULP of the CPU's.
  Added because a constant output (0 for tanh) is monotonic, bounded and
  symmetric: the three properties alone cannot tell it from tanh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

EXAMPLES = {
    f"{op}_{fmt}_props": Example(
        op, "WK-TRN-04", title, (SHAPE16,), SHAPE16, EXACT, fmt,
        init="bits16", check="sigtanh",
    )  # fmt: skip
    for op, title in (("sigmoid", "Sigmoid"), ("tanh", "Tanh"))
    for fmt in ("fp16", "bf16")
}

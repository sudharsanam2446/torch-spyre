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


"""LeakyReLU (WK-ACT-03, "Missing Tests" item 5: no LeakyReLU test).

The spec names LeakyReLU ("ReLU / LeakyReLU / Clip"), but torch_spyre has no
handler, lowering, decomposition or test for it. Inductor's default
decomposition gives ``where(x > 0, x, x * slope)`` and the pieces of that exist,
so it may compile by accident, or it may not. Nobody has run it: a compile
failure here is the finding, not a test bug.

Four examples, ``leaky_relu_<fmt>_<slope>``: fp16 and bf16, slopes 0.01 (the
PyTorch default) and 0.2. The input holds every 16-bit pattern once. Criteria
(``leakyrelu``): positive branch bit-exact; negative branch within 1 ULP of
``x * slope``, normal and subnormal results as separate checks, so a device that
flushes subnormals shows up under its own name; zeros keep their sign; inf and
NaN are kept.

The slope is a compile-time constant, so each slope is a different kernel. If it
becomes an extra kernel argument, or the op compiles to more than one kernel, the
build fails ("expected 1 kernel") or ``spyre_cli.launch`` raises; either is
reported.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

EXAMPLES = {
    f"leaky_relu_{fmt}_{slope}": Example(
        "leaky_relu", "WK-ACT-03", f"LeakyReLU (slope {slope})", (SHAPE16,), SHAPE16,
        EXACT, fmt, init="bits16", params={"slope": slope}, check="leakyrelu",
    )  # fmt: skip
    for fmt in ("fp16", "bf16")
    for slope in (0.01, 0.2)
}

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


"""Bit-exact clamp / clip (WK-ACT-03, "Missing Tests" item 7).

Two examples, ``clamp_fp16_exact`` and ``clamp_bf16_exact``, each compiling
``torch.clamp(x, min=-1.5, max=3.5)``. Both bounds are exact in fp16 and bf16
(-1.5 = -0b1.1, 3.5 = 0b11.1), so no rounding of the bounds is involved. The
input holds every 16-bit pattern once and the output is compared bit for bit with
PyTorch's CPU clamp. Criteria (``clamp``): inside (bits unchanged), below / above
(saturate to the bound), edges (each bound and one step either side, where a
``<`` against ``<=`` shows up), special (sign of zero, NaN).

Whether the bundle can be launched standalone is part of what is tested. For
clamp, Inductor keeps the bounds out of the kernel argument list (they become
``clipMin`` / ``clipMax`` constants of the kernel), so the bundle should take
only ``(x, out)``. If it takes more, ``spyre_cli.launch`` raises and the check
reports it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

CLAMP_MIN = -1.5
CLAMP_MAX = 3.5

EXAMPLES = {
    f"clamp_{fmt}_exact": Example(
        "clamp", "WK-ACT-03", "Clamp (bit-exact)", (SHAPE16,), SHAPE16, EXACT, fmt,
        init="bits16", params={"lo": CLAMP_MIN, "hi": CLAMP_MAX}, check="clamp",
    )  # fmt: skip
    for fmt in ("fp16", "bf16")
}

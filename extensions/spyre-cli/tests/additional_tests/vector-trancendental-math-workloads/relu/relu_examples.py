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


"""Bit-exact ReLU (WK-ACT-03, "Missing Tests" item 7: no bit-exact comparison).

The spec's metric for WK-ACT-03 is *bit-exact threshold switching*, which a
tolerance cannot show: on fp16 values below 4 in magnitude, a ReLU that flushes
|x| < 0.09 to zero passes the 0.1 default, and ``assert_close(+0.0, -0.0)``
passes too, so even an exact tolerance cannot see the sign of zero.

Two examples, ``relu_fp16_exact`` and ``relu_bf16_exact``. The input holds every
16-bit pattern once, so the sweep is exhaustive and the comparison is bit for
bit against PyTorch's CPU ``torch.relu``. Criteria (``relu``): positive (output
bits == input bits), negative (+0.0), zero (sign kept), inf/nan.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import EXACT, SHAPE16, Example  # noqa: E402

EXAMPLES = {
    f"relu_{fmt}_exact": Example(
        "relu", "WK-ACT-03", "ReLU (bit-exact)", (SHAPE16,), SHAPE16, EXACT, fmt,
        init="bits16", check="relu",
    )  # fmt: skip
    for fmt in ("fp16", "bf16")
}

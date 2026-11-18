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


"""BF16 spyre-cli examples for the Vector & Transcendental Math workloads.

Covers WK-TRN-01..04 and WK-ACT-01..02. The spec lists BF16 for all of them,
but the only BF16 tests in the repo were attention shapes, scalars, conversion
and transpose. Each example here pushes one transcendental / activation op
through ``torch.compile`` and ``spyre launch`` in bf16, over the input domain
the spec states:

    ======================  ===========  =================================
    example                 workload     input domain
    ======================  ===========  =================================
    exp_bf16                WK-TRN-01    N(0, 1)
    exp_wide_bf16           WK-TRN-01    U(-88, 88)   (spec domain)
    rsqrt_bf16              WK-TRN-02    10**U(-7, 4) (spec domain)
    reciprocal_bf16         WK-TRN-03    |N(0, 1)| + 0.5
    div_bf16                WK-TRN-03    |N(0, 1)| + 0.5
    sigmoid_bf16            WK-TRN-04    U(-16, 16)   (spec domain)
    tanh_bf16               WK-TRN-04    U(-16, 16)   (spec domain)
    gelu_bf16               WK-ACT-01    U(-10, 10)   (spec domain, erf)
    gelu_tanh_bf16          WK-ACT-01    U(-10, 10)   (spec domain, tanh)
    silu_bf16               WK-ACT-02    U(-10, 10)   (spec domain)
    swiglu_bf16             WK-ACT-02    N(0, 1)
    ======================  ===========  =================================

torch_spyre/csrc/types_mapping.h maps bfloat16 to the device FP16 format
(SEN169_FP16), so a bf16 value outside the FP16 range (e.g. exp_wide_bf16) is
exactly the case these examples are meant to exercise: a mismatch against the
CPU reference is a real finding, not a test bug.

Importing this module does not import torch, so tests can be collected without
a Spyre device. It only registers the examples; building and checking them is done
by the upstream examples/examples.py and examples/check.py, driven by
tests/additional_tests/base/run_all.py:

    python base/run_all.py bf16            # this suite, SDK check only

Examples that compile to more than one kernel are rejected, since one kernel is
launched per example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import Example  # noqa: E402

# (atol, rtol). bf16 keeps 8 significant bits (relative step 2**-8 ~= 0.4%), so
# a one-ulp difference between the device and the fp32 CPU reference can reach
# ~0.8%. 2% leaves headroom for that rounding and no more.
BF16_TOL = (0.02, 0.02)

_M = (512, 1024)
_D = ((256, 2048), (256, 2048))
_S = ((128, 4096), (128, 4096))


def _ex(op, workload, title, shapes=(_M,), output=_M, **kw):
    return Example(op, workload, title, shapes, output, BF16_TOL, "bf16", **kw)


EXAMPLES = {
    # WK-TRN-01: exp. The spec domain is [-88, 88]; exp overflows fp16 above
    # 11.09, so exp_wide_bf16 only passes if bf16 really has the fp32 range.
    "exp_bf16": _ex("exp", "WK-TRN-01", "Exponential"),
    "exp_wide_bf16": _ex(
        "exp", "WK-TRN-01", "Exponential - wide range", init="uniform:-88:88"
    ),
    # WK-TRN-02: rsqrt over x in [1e-7, 1e4]. The eps = 1e-6 guard is a scalar
    # constant, which spyre launch cannot supply (see README, Limitations), so
    # the guard is covered by tests/inductor/test_wk_trn_02_rsqrt.py instead.
    "rsqrt_bf16": _ex(
        "rsqrt", "WK-TRN-02", "Reciprocal square root", init="logspace:-7:4"
    ),
    # WK-TRN-03: reciprocal and real division
    "reciprocal_bf16": _ex("reciprocal", "WK-TRN-03", "Reciprocal", init="positive"),
    "div_bf16": _ex("div", "WK-TRN-03", "Division", _D, _D[0], init="positive"),
    # WK-TRN-04: sigmoid and tanh over x in [-16, 16]
    "sigmoid_bf16": _ex("sigmoid", "WK-TRN-04", "Sigmoid", init="uniform:-16:16"),
    "tanh_bf16": _ex("tanh", "WK-TRN-04", "Tanh", init="uniform:-16:16"),
    # WK-ACT-01: GELU, exact (erf) and tanh approximation, x in [-10, 10]
    "gelu_bf16": _ex("gelu", "WK-ACT-01", "GELU", init="uniform:-10:10"),
    "gelu_tanh_bf16": _ex(
        "gelu_tanh", "WK-ACT-01", "GELU tanh approximation", init="uniform:-10:10"
    ),
    # WK-ACT-02: SiLU over x in [-10, 10], and the fused SwiGLU
    "silu_bf16": _ex("silu", "WK-ACT-02", "SiLU", init="uniform:-10:10"),
    "swiglu_bf16": _ex("swiglu", "WK-ACT-02", "SwiGLU", _S, _S[0]),
}

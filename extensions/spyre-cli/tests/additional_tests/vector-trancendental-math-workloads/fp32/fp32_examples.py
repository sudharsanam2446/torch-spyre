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


"""FP32 spyre-cli examples for the Vector & Transcendental Math workloads.

Closes item 4 of the "Missing Tests" sheet ("No FP32 test for exp, sigmoid,
reciprocal, ..."), for the workloads whose spec lists FP32 and whose op has an
FP32 kernel:

    ===========================  ===========  =================================
    example                      workload     input domain
    ===========================  ===========  =================================
    exp_fp32                     WK-TRN-01    N(0, 1)
    exp_wide_fp32                WK-TRN-01    U(-88, 88)   (spec domain)
    reciprocal_fp32              WK-TRN-03    |N(0, 1)| + 0.5
    reciprocal_wide_fp32         WK-TRN-03    10**U(-30, 30)
    reciprocal_subnormal_fp32    WK-TRN-03    10**U(-44, -38)  (FP32 subnormals)
    ===========================  ===========  =================================

WK-TRN-02 (rsqrt) and FP32 division are not repeated here: both already have
an FP32 test in the "Test Cases" sheet (VT-E06 ``test_rsqrt_fp32``, VT-E09
``test_pointwise_binary_op_fp32``). Of the ops named in item 4, only exp and
reciprocal belong to WK-TRN-01..03; sigmoid, tanh, SiLU, GELU and clamp are
WK-TRN-04 / WK-ACT-* and are outside this folder.

Findings to expect, not test bugs:

- ``exp_wide_fp32``: exp overflows FP32 above x = 88.72, so the top of the spec
  domain (88) is the last finite point; results below 1.18e-38 are subnormal
  and a device that flushes denormals returns 0 for them (hence the absolute
  tolerance, which only forgives that).
- ``reciprocal_subnormal_fp32``: 1/x overflows FP32 for x < 2.94e-39, so the
  reference is inf for part of the sweep and the device must return inf too.
  If it flushes subnormal inputs to zero it returns inf for all of them and the
  finite tail is a mismatch.

Importing this module does not import torch, so tests can be collected without
a Spyre device. It only registers the examples; building and checking them is done
by the upstream examples/examples.py and examples/check.py, driven by
tests/additional_tests/base/run_all.py:

    python base/run_all.py fp32            # this suite, SDK check only

Examples that compile to more than one kernel are rejected, since one kernel is
launched per example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base"))

from workloads_common import Example  # noqa: E402

# (atol, rtol) against a float64 reference rounded to fp32. FP32 keeps 24
# significant bits (relative step 1.2e-7); 1e-5 is the rtol that
# tests/inductor/test_vector_transcendental_workloads.py uses for FP32 exp. The
# absolute term only forgives subnormal results (below 1.18e-38) that a
# denormal-flushing device turns into 0.
FP32_TOL = (1e-37, 1e-5)

_M = (512, 1024)


def _ex(op, workload, title, **kw):
    # the reference is computed in float64: see workloads_common.expected
    return Example(op, workload, title, (_M,), _M, FP32_TOL, "fp32", **kw)


EXAMPLES = {
    # WK-TRN-01: exp. FP32 does not overflow until 88.72, so unlike FP16 the
    # whole spec domain [-88, 88] is meaningful.
    "exp_fp32": _ex("exp", "WK-TRN-01", "Exponential"),
    "exp_wide_fp32": _ex(
        "exp", "WK-TRN-01", "Exponential - wide range", init="uniform:-88:88"
    ),
    # WK-TRN-03: reciprocal, from ordinary values down to FP32 subnormals.
    "reciprocal_fp32": _ex("reciprocal", "WK-TRN-03", "Reciprocal", init="positive"),
    "reciprocal_wide_fp32": _ex(
        "reciprocal", "WK-TRN-03", "Reciprocal - wide range", init="logspace:-30:30"
    ),
    "reciprocal_subnormal_fp32": _ex(
        "reciprocal", "WK-TRN-03", "Reciprocal - subnormal", init="logspace:-44:-38"
    ),
}

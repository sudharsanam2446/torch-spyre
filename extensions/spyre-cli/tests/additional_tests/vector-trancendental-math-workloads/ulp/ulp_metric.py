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


"""ULP-error measurement ("Missing Tests" item 1: no ULP measurement).

The spec's main accuracy metric is ULP (units in the last place), but the
repo's tests only use allclose tolerances. For a result ``got`` and the exact
value ``ref`` (float64):

    ulp_error = |got - ref| / spacing(ref)

``spacing(ref)`` is the gap between the two representable numbers of the format
around ``ref`` (Goldberg's ulp), fixed at the subnormal spacing below the
smallest normal. A correctly rounded result has ulp_error <= 0.5, so the spec's
"Max ULP error <= 1.5" leaves one ULP for the implementation.

Points are sorted into classes:

* normal   -- |ref| >= the smallest normal of the format. The ULP limit applies
              here.
* tiny     -- zero or subnormal results, reported apart: a device that flushes
              subnormals to zero is a legitimate design and would otherwise
              swamp the normal-range figure.
* special  -- ref is NaN, or rounds to +-inf in the format (overflow, 1/0, inf
              inputs). The device must return the same class, or it is a
              ``class_mismatch``.

Needs only torch, so the metric can be checked on the CPU.
"""

import dataclasses
import math

import torch

from workloads_common import ops


@dataclasses.dataclass(frozen=True)
class Fmt:
    name: str
    dtype: torch.dtype
    int_dtype: torch.dtype
    width: int  # total bits
    mant: int  # explicit mantissa bits
    emin: int  # exponent of the smallest normal number


FORMATS = {
    "fp16": Fmt("fp16", torch.float16, torch.int16, 16, 10, -14),
    "bf16": Fmt("bf16", torch.bfloat16, torch.int16, 16, 7, -126),
    "fp32": Fmt("fp32", torch.float32, torch.int32, 32, 23, -126),
}


def _gelu_exact(x):
    # 0.5 * x * (1 + erf(x / sqrt 2)) cancels to 0 below x = -6, even in float64
    # (torch's gelu gives -0.0 at x = -9, true value -1e-18). erfc does not cancel.
    return 0.5 * x * torch.erfc(-x / math.sqrt(2.0))


def _gelu_tanh(x):
    # 1 + tanh(u) = 2 / (1 + exp(-2u)), which avoids the same cancellation.
    u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    return x / (1.0 + torch.exp(-2.0 * u))


# Float64 references where torch's own float64 op is not accurate enough.
REFERENCES = {"gelu": _gelu_exact, "gelu_tanh": _gelu_tanh}


def reference(op, inputs):
    """The exact result of ``op`` in float64."""
    fn = REFERENCES.get(op, ops()[op])
    return fn(*[t.to(torch.float64) for t in inputs])


def spacing(ref, fmt):
    """ulp(ref) in ``fmt`` for each element of the float64 tensor ``ref``."""
    exponent = torch.frexp(ref.abs()).exponent.to(torch.int64) - 1
    exponent = exponent.clamp(min=fmt.emin)
    return torch.exp2((exponent - fmt.mant).to(torch.float64))


def _stats(err):
    if err.numel() == 0:
        return {"n": 0, "max": 0.0, "mean": 0.0, "p99": 0.0}
    return {
        "n": int(err.numel()),
        "max": float(err.max()),
        "mean": float(err.mean()),
        "p99": float(torch.quantile(err, 0.99)),
    }


def _point(inputs, got64, want, index, ulp=None):
    point = {
        "inputs": [float(t.reshape(-1)[index].to(torch.float64)) for t in inputs],
        "got": float(got64[index]),
        "want": float(want[index]),
    }
    if ulp is not None:
        point["ulp"] = float(ulp[index])
    return point


def measure(inputs, got, ref, fmt_name):
    """ULP statistics of ``got`` against the float64 reference ``ref``."""
    fmt = FORMATS[fmt_name]
    got = got.detach().cpu().reshape(-1).to(fmt.dtype)
    got64 = got.to(torch.float64)
    ref = ref.detach().cpu().reshape(-1).to(torch.float64)
    ref_r = ref.to(fmt.dtype).to(torch.float64)  # correctly rounded reference

    exp_nan = torch.isnan(ref)
    exp_inf = torch.isinf(ref_r)
    got_nan = torch.isnan(got64)
    got_inf = torch.isinf(got64)
    class_ok = torch.where(
        exp_nan,
        got_nan,
        torch.where(exp_inf, got64 == ref_r, ~(got_nan | got_inf)),
    )
    finite = class_ok & ~exp_nan & ~exp_inf

    err = torch.zeros_like(got64)
    err[finite] = (got64[finite] - ref[finite]).abs() / spacing(ref[finite], fmt)
    normal = finite & (ref.abs() >= 2.0**fmt.emin)
    tiny = finite & ~normal

    report = {
        "format": fmt_name,
        "n": int(ref.numel()),
        "n_special": int((exp_nan | exp_inf).sum()),
        "class_mismatch": int((~class_ok).sum()),
        "normal": _stats(err[normal]),
        "tiny": _stats(err[tiny]),
    }
    for label, mask in (("worst_normal", normal), ("worst_tiny", tiny)):
        if mask.any():
            masked = torch.where(mask, err, torch.full_like(err, -1.0))
            report[label] = _point(inputs, got64, ref_r, int(masked.argmax()), err)
    bad = (~class_ok).nonzero().reshape(-1)[:3]
    report["class_mismatch_examples"] = [
        _point(inputs, got64, ref_r, int(i)) for i in bad
    ]
    return report


def verdict(report, limit):
    """PASS / FAIL against ``limit``, or MEASURED when there is no limit.

    The limit applies to normal results. A limit of 0 means bit-exact, so tiny
    (subnormal) results are gated too. Any class mismatch is a FAIL.
    """
    if limit is None:
        return "MEASURED"
    worst = report["normal"]["max"]
    if limit == 0:
        worst = max(worst, report["tiny"]["max"])
    if report["class_mismatch"] > 0 or worst > limit:
        return "FAIL"
    return "PASS"


def _fmt_point(point):
    xs = ",".join(f"{v:.6g}" for v in point["inputs"])
    text = f"x={xs} got={point['got']:.6g} want={point['want']:.6g}"
    if "ulp" in point:
        text += f" ulp={point['ulp']:.3f}"
    return text


def one_line(report):
    """The figures of ``report`` on one line, with the worst points."""
    text = (
        f"max_ulp={report['normal']['max']:.3f} mean={report['normal']['mean']:.3f} "
        f"p99={report['normal']['p99']:.3f} tiny_max={report['tiny']['max']:.3f} "
        f"class_mismatch={report['class_mismatch']} n={report['n']}"
    )
    if "worst_normal" in report:
        text += " worst(" + _fmt_point(report["worst_normal"]) + ")"
    for point in report["class_mismatch_examples"][:1]:
        text += " class-mismatch(" + _fmt_point(point) + ")"
    return text

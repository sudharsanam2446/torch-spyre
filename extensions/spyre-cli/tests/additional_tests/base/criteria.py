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


"""Pass criteria of the suites that compare more than ``allclose``.

``allclose`` with atol = rtol = 0.1 cannot show a flushed subnormal, a lost
sign of zero, a kink of a few ULP, or an eps guard that does nothing. These
criteria can. ``criteria_for(ex)`` returns ``label -> criterion`` for an
example, where a criterion is ``f(inputs, got)`` returning ``None`` on success
or a message naming the first offending points. ``inputs`` are the CPU inputs
the kernel was given, ``got`` the CPU copy of the device output.

``Example.check`` selects the set:

    relu       WK-ACT-03  bit-exact ReLU: positive / negative / zero / inf-nan
    clamp      WK-ACT-03  bit-exact clamp: inside / below / above / edges / special
    leakyrelu  WK-ACT-03  LeakyReLU: positive / neg / subnorm / zero / inf-nan
    sigtanh    WK-TRN-04  sigmoid / tanh: monotonic / bounded / symmetry / anchors
    eps        WK-TRN-02  rsqrt(x + eps): zero / subnormal / small / large
    edge       WK-TRN-03  reciprocal, div and scalar ops on zeros, subnormals,
                          infinities and NaN, against the CPU result
    ulp        item 1     handled by check.py (a measurement, not a criterion)

Only torch is needed, so every criterion can be tested on the CPU (see
``test_workloads.py``): it accepts the CPU reference and rejects the fault it
exists for.
"""

import torch

from workloads_common import build_op, torch_dtype

# Note for reviewers: why this file exists.
#
# Upstream examples/check.py has one pass rule: torch.allclose(got, want, atol,
# rtol). That cannot express most of what the "Missing Tests" sheet asks for
# (bit-exact ReLU / clamp, flushed subnormals, sign of zero, monotonicity,
# symmetry, NaN / inf handling, an eps guard that does nothing).
#
# This file holds the named pass/fail rules the sheet needs, one family per
# suite (relu, clamp, leakyrelu, sigtanh, eps, edge). A failure is reported per
# class ("zero failed"), where a single allclose would give one yes/no for the
# whole tensor. The bf16 and fp32 suites do not use this file: they run
# upstream's own allclose check unmodified.

# --------------------------------------------------------------------------
# Helpers for 16-bit formats
# --------------------------------------------------------------------------


def bits_of(t):
    return t.reshape(-1).view(torch.int16)


def patterns(t):
    """The 16-bit patterns of ``t`` as non-negative ints."""
    return bits_of(t).to(torch.int32) & 0xFFFF


def ordered(t):
    """Integer that increases with the value; differences are distances in ULPs."""
    p = patterns(t)
    mag = p & 0x7FFF
    return torch.where((p & 0x8000) != 0, -mag, mag)


def _describe(inputs, got, want, bad, count=5):
    """The first few offending points, with the bits of every operand."""
    index = bad.nonzero().flatten()[:count].tolist()
    flat = [t.reshape(-1) for t in inputs]
    g, w = got.reshape(-1), want.reshape(-1)

    def show(t, i):
        return f"{t[i].double().item():.6g} (0x{int(bits_of(t)[i]) & 0xFFFF:04x})"

    names = ("x",) if len(flat) == 1 else ("a", "b")
    rows = [
        " ".join(f"{n}={show(t, i)}" for n, t in zip(names, flat))
        + f" got={show(g, i)} want={show(w, i)}"
        for i in index
    ]
    return f"{int(bad.sum())} points differ, e.g. " + "; ".join(rows)


def _message(inputs, got, want, bad):
    return _describe(inputs, got, want, bad) if bad.any() else None


# --------------------------------------------------------------------------
# relu: bit-exact
# --------------------------------------------------------------------------


def relu_criteria(ex):
    def reference(x):
        return torch.relu(x)

    def positive(inputs, got):
        """x > 0 (incl. +inf and positive subnormals): output bits == input bits."""
        (x,) = inputs
        mask = x.reshape(-1) > 0
        return _message(inputs, got, x, mask & (bits_of(got) != bits_of(x)))

    def negative(inputs, got):
        """x < 0, finite: output bits == 0 (+0.0)."""
        (x,) = inputs
        xf = x.reshape(-1)
        mask = (xf < 0) & xf.isfinite()
        return _message(inputs, got, reference(x), mask & (bits_of(got) != 0))

    def zero(inputs, got):
        """+0.0 -> +0.0 and -0.0 -> -0.0, bit for bit, as on the CPU."""
        (x,) = inputs
        want = reference(x)
        mask = x.reshape(-1) == 0
        return _message(inputs, got, want, mask & (bits_of(got) != bits_of(want)))

    def inf_nan(inputs, got):
        """+inf -> +inf, -inf -> +0.0, NaN -> NaN."""
        (x,) = inputs
        want, xf = reference(x), x.reshape(-1)
        problems = []
        plus, minus = xf == float("inf"), xf == float("-inf")
        if (bits_of(got)[plus] != bits_of(want)[plus]).any():
            problems.append("+inf did not stay +inf")
        if (bits_of(got)[minus] != bits_of(want)[minus]).any():
            problems.append("-inf did not become +0.0")
        if not got.reshape(-1)[xf.isnan()].isnan().all():
            problems.append("NaN did not stay NaN")
        return "; ".join(problems) or None

    return {"positive": positive, "negative": negative, "zero": zero, "inf/nan": inf_nan}


# --------------------------------------------------------------------------
# clamp: bit-exact, with the bounds of the example
# --------------------------------------------------------------------------


def clamp_criteria(ex):
    lo, hi = ex.params["lo"], ex.params["hi"]

    def reference(x):
        return torch.clamp(x, min=lo, max=hi)

    def edge_mask(x):
        """The bounds and the value one step (one bit pattern) either side."""
        xb = bits_of(x).to(torch.int32)
        mask = torch.zeros_like(xb, dtype=torch.bool)
        for bound in (lo, hi):
            at = int(torch.tensor(bound, dtype=x.dtype).view(torch.int16))
            mask |= (xb == at) | (xb == at + 1) | (xb == at - 1)
        return mask & x.reshape(-1).isfinite()

    def inside(inputs, got):
        """min <= x <= max: output bits == input bits."""
        (x,) = inputs
        xf = x.reshape(-1)
        mask = (xf >= lo) & (xf <= hi)
        return _message(inputs, got, x, mask & (bits_of(got) != bits_of(x)))

    def below(inputs, got):
        """x < min (incl. -inf): output bits == the bits of min."""
        (x,) = inputs
        want = reference(x)
        mask = x.reshape(-1) < lo
        return _message(inputs, got, want, mask & (bits_of(got) != bits_of(want)))

    def above(inputs, got):
        """x > max (incl. +inf): output bits == the bits of max."""
        (x,) = inputs
        want = reference(x)
        mask = x.reshape(-1) > hi
        return _message(inputs, got, want, mask & (bits_of(got) != bits_of(want)))

    def edges(inputs, got):
        """The bounds and one step either side: where an off-by-one shows up."""
        (x,) = inputs
        want = reference(x)
        bad = edge_mask(x) & (bits_of(got) != bits_of(want))
        return _message(inputs, got, want, bad)

    def special(inputs, got):
        """+0.0 -> +0.0, -0.0 -> -0.0 (bit for bit), NaN stays NaN."""
        (x,) = inputs
        want, xf = reference(x), x.reshape(-1)
        zero = xf == 0
        problems = []
        if (bits_of(got)[zero] != bits_of(want)[zero]).any():
            problems.append("a zero changed its bits (sign of zero lost)")
        if not got.reshape(-1)[xf.isnan()].isnan().all():
            problems.append("NaN did not stay NaN")
        return "; ".join(problems) or None

    return {
        "inside": inside,
        "below": below,
        "above": above,
        "edges": edges,
        "special": special,
    }


# --------------------------------------------------------------------------
# leaky_relu: bit-exact positive branch, 1 ULP on the negative branch
# --------------------------------------------------------------------------


def leakyrelu_criteria(ex):
    slope = ex.params["slope"]

    def reference(x):
        """x for x > 0, else x * slope, rounded once. The slope is first rounded
        to the dtype, since that is the constant a 16-bit kernel multiplies by."""
        held = float(torch.tensor(slope, dtype=x.dtype))
        scaled = (x.double() * held).to(x.dtype)
        return torch.where(x > 0, x, scaled)

    def positive(inputs, got):
        """x > 0 (incl. +inf and positive subnormals): output bits == input bits."""
        (x,) = inputs
        mask = x.reshape(-1) > 0
        return _message(inputs, got, x, mask & (bits_of(got) != bits_of(x)))

    def negative(subnormal):
        def check(inputs, got):
            (x,) = inputs
            want = reference(x).reshape(-1)
            xf = x.reshape(-1)
            small = want.double().abs() < torch.finfo(x.dtype).tiny
            mask = (xf < 0) & xf.isfinite() & (small if subnormal else ~small)
            bad = mask & ((ordered(got) - ordered(want)).abs() > 1)
            return _message(inputs, got, want, bad)

        check.__doc__ = "x < 0 with a %s result: within 1 ULP of x * slope." % (
            "subnormal or underflowing" if subnormal else "normal"
        )
        return check

    def zero(inputs, got):
        """+0.0 -> +0.0 and -0.0 -> -0.0, bit for bit."""
        (x,) = inputs
        want = reference(x)
        mask = x.reshape(-1) == 0
        return _message(inputs, got, want, mask & (bits_of(got) != bits_of(want)))

    def inf_nan(inputs, got):
        """+inf -> +inf, -inf -> -inf, NaN -> NaN."""
        (x,) = inputs
        xf, gf = x.reshape(-1), got.reshape(-1)
        problems = []
        if not (gf[xf == float("inf")] == float("inf")).all():
            problems.append("+inf did not stay +inf")
        if not (gf[xf == float("-inf")] == float("-inf")).all():
            problems.append("-inf did not stay -inf")
        if not gf[xf.isnan()].isnan().all():
            problems.append("NaN did not stay NaN")
        return "; ".join(problems) or None

    return {
        "positive": positive,
        "neg": negative(False),
        "subnorm": negative(True),  # a device that flushes subnormals fails only here
        "zero": zero,
        "inf/nan": inf_nan,
    }


# --------------------------------------------------------------------------
# sigmoid / tanh: monotonicity, bounds, symmetry and anchors over [-16, 16]
# --------------------------------------------------------------------------

DOMAIN = 16.0


def _domain_mask(x):
    """Finite inputs with |x| <= 16, the spec domain."""
    xf = x.reshape(-1)
    return xf.isfinite() & (xf.abs() <= DOMAIN)


def _partner_index(x):
    """For each element of the sweep, the index of the element holding -x."""
    p = patterns(x)
    position = torch.empty(65536, dtype=torch.long)
    position[p.long()] = torch.arange(p.numel())
    return position[(p ^ 0x8000).long()]


def sigtanh_criteria(ex):
    fn = ex.op
    low = 0.0 if fn == "sigmoid" else -1.0

    def monotonic(inputs, got):
        """Sorted by x over the domain, the output never goes down (non-strict).

        Strict monotonicity cannot hold: both functions round to exactly 1.0
        well inside the domain.
        """
        (x,) = inputs
        keep = _domain_mask(x) & (patterns(x) != 0x8000)  # -0.0 ties with +0.0
        xs = x.reshape(-1)[keep].double()
        ys = got.reshape(-1)[keep].double()
        xp = patterns(x)[keep]
        order = torch.sort(xs, stable=True).indices
        xs, ys, xp = xs[order], ys[order], xp[order]
        down = (ys[1:] < ys[:-1]).nonzero().flatten()
        if down.numel() == 0:
            return None
        rows = [
            f"x={xs[i]:.6g} (0x{int(xp[i]):04x}) -> {ys[i]:.6g}, then "
            f"x={xs[i + 1]:.6g} -> {ys[i + 1]:.6g}"
            for i in down[:5].tolist()
        ]
        return f"{down.numel()} steps go down, e.g. " + "; ".join(rows)

    def bounded(inputs, got):
        """sigmoid in [0, 1], tanh in [-1, 1], no NaN, over the domain."""
        (x,) = inputs
        xs, ys = x.reshape(-1).double(), got.reshape(-1).double()
        bad = _domain_mask(x) & (ys.isnan() | (ys < low) | (ys > 1.0))
        if not bad.any():
            return None
        rows = [
            f"x={xs[i]:.6g} -> {ys[i]:.6g}" for i in bad.nonzero().flatten()[:5].tolist()
        ]
        return (
            f"{int(bad.sum())} points fail, e.g. " + "; ".join(rows)
            + f" (allowed range [{low:g}, 1])"
        )

    def tanh_symmetry(inputs, got):
        """tanh(-x) = -tanh(x) within 1 ULP, over the domain."""
        (x,) = inputs
        steps, partner = ordered(got), _partner_index(x)
        distance = (steps + steps[partner]).abs()
        bad = _domain_mask(x) & (distance > 1)
        if not bad.any():
            return None
        xs, ys = x.reshape(-1).double(), got.reshape(-1).double()
        rows = [
            f"x={xs[i]:.6g}: f(x)={ys[i]:.6g}, f(-x)={ys[partner[i]]:.6g}, "
            f"{int(distance[i])} ULP apart"
            for i in bad.nonzero().flatten()[:5].tolist()
        ]
        return f"{int(bad.sum())} points fail, e.g. " + "; ".join(rows) + " (allowed: 1 ULP)"

    def sigmoid_symmetry(inputs, got):
        """sigmoid(x) + sigmoid(-x) = 1 within 1 ULP of 1 (eps/2), over the domain."""
        (x,) = inputs
        tolerance = float(torch.finfo(x.dtype).eps) / 2
        ys, partner = got.reshape(-1).double(), _partner_index(x)
        error = (ys + ys[partner] - 1.0).abs()
        bad = _domain_mask(x) & ~(error <= tolerance)  # NaN counts as bad
        if not bad.any():
            return None
        xs = x.reshape(-1).double()
        rows = [
            f"x={xs[i]:.6g}: f(x)+f(-x)={ys[i] + ys[partner[i]]:.6g}"
            for i in bad.nonzero().flatten()[:5].tolist()
        ]
        return (
            f"{int(bad.sum())} points fail, e.g. " + "; ".join(rows)
            + f" (allowed: |sum - 1| <= {tolerance:g})"
        )

    def anchors(inputs, got):
        """At x = 0, +-1, +-4 and +-16 the result is within 1 ULP of the CPU's.

        Monotonic, bounded and symmetric are all true of a constant output (0 for
        tanh, 0.5 for sigmoid); the anchors pin the function down.
        """
        (x,) = inputs
        want = build_op(ex)(x)
        at = torch.zeros_like(x, dtype=torch.bool).reshape(-1)
        for point in (0.0, 1.0, -1.0, 4.0, -4.0, 16.0, -16.0):
            at |= x.reshape(-1) == point
        bad = at & ((ordered(got) - ordered(want)).abs() > 1)
        return _message(inputs, got, want, bad)

    return {
        "monotonic": monotonic,
        "bounded": bounded,
        "symmetry": tanh_symmetry if fn == "tanh" else sigmoid_symmetry,
        "anchors": anchors,
    }


# --------------------------------------------------------------------------
# rsqrt(x + eps): the guard keeps the result finite
# --------------------------------------------------------------------------

EPS_RTOL = {"fp32": 1e-4, "fp16": 2e-2, "bf16": 2e-2}  # see README: the spec's 1e-4
SMALL_LIMIT = 1e-3  # above this, eps changes the sum by less than 0.05% of x


def eps_criteria(ex):
    eps = ex.params["eps"]
    rtol = EPS_RTOL[ex.dtype]

    def expected(inputs):
        """The exact ``rsqrt(x + eps)`` (float64, flat) for the stored inputs."""
        return torch.rsqrt(inputs[0].reshape(-1).double() + eps)

    def masks(inputs):
        x = inputs[0].reshape(-1).double()
        tiny = torch.finfo(torch_dtype(ex)).tiny
        zero = x == 0
        sub = ~zero & (x.abs() < tiny)
        small = ~zero & ~sub & (x.abs() <= SMALL_LIMIT)
        return {"zero": zero, "subnormal": sub, "small": small, "large": x.abs() > SMALL_LIMIT}

    def criterion(label):
        def check(inputs, got):
            g, w = got.reshape(-1).double(), expected(inputs)
            wrong = ~g.isfinite() | ((g - w).abs() > rtol * w.abs())
            bad = masks(inputs)[label] & wrong
            if not bad.any():
                return None
            x = inputs[0].reshape(-1)
            rows = [
                f"x={x[i].double().item():.6g} got={g[i].item():.6g} want={w[i].item():.6g}"
                for i in bad.nonzero().flatten()[:5].tolist()
            ]
            return f"{int(bad.sum())} points differ, e.g. " + "; ".join(rows)

        check.__doc__ = f"{label} elements are within rtol of rsqrt(x + eps)."
        return check

    return {label: criterion(label) for label in ("zero", "subnormal", "small", "large")}


# --------------------------------------------------------------------------
# edge cases: reciprocal, div and scalar ops against the CPU result
# --------------------------------------------------------------------------

LIMIT_ULPS = 1  # for finite nonzero results; everything else must match bit for bit
RECIPROCAL_LABELS = ("zero", "inf_nan", "subnormal", "normal")  # also scalar ops
DIV_LABELS = ("zero_divisor", "non_finite", "tiny_divisor", "tiny_numerator", "normal")


def _classes(t):
    """Boolean masks (flat) of the kinds of value in ``t``."""
    f = t.reshape(-1)
    finite = f.isfinite()
    zero = f == 0
    sub = finite & ~zero & (f.double().abs() < torch.finfo(t.dtype).tiny)
    return {
        "nan": f.isnan(),
        "inf": f.isinf(),
        "zero": zero,
        "sub": sub,
        "normal": finite & ~zero & ~sub,
        "finite": finite,
    }


def _mismatch(got, want):
    """Flat mask of the elements where ``got`` breaks the pass rule: NaN stays
    NaN, an infinity or zero matches bit for bit, any other result is finite,
    nonzero and within 1 ULP."""
    g, w = got.reshape(-1), want.reshape(-1)
    gb, wb = bits_of(g), bits_of(w)
    w_nan = w.isnan()
    w_exact = w.isinf() | (w == 0)
    w_number = ~w_nan & ~w_exact
    bad = w_nan != g.isnan()
    bad |= w_exact & (gb != wb)
    distance = (ordered(g) - ordered(w)).abs()
    bad |= w_number & (~g.isfinite() | (g == 0) | (distance > LIMIT_ULPS))
    return bad


def edge_references(ex):
    """The CPU functions whose result a device result may match.

    ``build_op(ex)`` first. A scalar example adds a second one with the scalar
    first rounded to the tensor dtype, which is what a kernel holding its
    constant in 16 bits computes. A device element passes if it matches either.
    """
    fn = build_op(ex)
    if ex.op != "scalar_binary":
        return [fn]
    binary = {"mul": torch.mul, "div": torch.div, "add": torch.add}[ex.params["kind"]]
    constant = torch.tensor(ex.params["scalar"], dtype=torch_dtype(ex))
    return [fn, lambda x: binary(x, constant)]


def edge_labels(ex):
    return DIV_LABELS if ex.op == "div" else RECIPROCAL_LABELS


def _edge_masks(ex, inputs):
    if ex.op != "div":
        c = _classes(inputs[0])
        return {
            "zero": c["zero"],
            "inf_nan": c["inf"] | c["nan"],
            "subnormal": c["sub"],
            "normal": c["normal"],
        }
    a, b = _classes(inputs[0]), _classes(inputs[1])
    b_zero = b["zero"]
    return {
        "zero_divisor": b_zero,
        "non_finite": ~b_zero & (a["nan"] | a["inf"] | b["nan"] | b["inf"]),
        "tiny_divisor": b["sub"] & a["finite"],
        "tiny_numerator": a["sub"] & b["normal"],
        "normal": (a["normal"] | a["zero"]) & b["normal"],
    }


def edge_criteria(ex):
    def criterion(label):
        def check(inputs, got):
            wants = [fn(*inputs) for fn in edge_references(ex)]
            bad = _edge_masks(ex, inputs)[label]
            for want in wants:  # an element passes if it matches any reference
                bad = bad & _mismatch(got, want)
            return _describe(inputs, got, wants[0], bad) if bad.any() else None

        check.__doc__ = f"{label} elements match the CPU result."
        return check

    return {label: criterion(label) for label in edge_labels(ex)}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

CRITERIA = {
    "relu": relu_criteria,
    "clamp": clamp_criteria,
    "leakyrelu": leakyrelu_criteria,
    "sigtanh": sigtanh_criteria,
    "eps": eps_criteria,
    "edge": edge_criteria,
}


def criteria_for(ex):
    """``label -> criterion`` for example ``ex`` (``ex.check`` names the set)."""
    return CRITERIA[ex.check](ex)


def evaluate(ex, inputs, got):
    """Apply every criterion: ``(ok, "label=PASS|FAIL ... first failure (...)")``."""
    verdicts, first_failure = [], ""
    for label, criterion in criteria_for(ex).items():
        message = criterion(inputs, got)
        verdicts.append(f"{label}={'PASS' if message is None else 'FAIL'}")
        if message is not None and not first_failure:
            first_failure = f" first failure ({label}): {message}"
    return not first_failure, " ".join(verdicts) + first_failure


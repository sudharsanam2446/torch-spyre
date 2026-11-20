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


"""Input sweeps for the suites that check more than ``allclose``.

``Example.init`` selects one (``workloads_common.make_inputs`` calls ``build``):

    bits16        every 16-bit pattern once (fp16 / bf16, 65536 elements): so
                  every zero, subnormal, normal, infinity and NaN is an input
    sweep=SPEC    ULP sweeps, comma separated tokens each filling an equal share
                  of the tensor:
                    bits:LO:HI    every representable value in [LO, HI] in bit
                                  order, so each binade has equal weight.
                                  Exhaustive when the format has fewer values
                                  than the tensor has elements (fp16, bf16),
                                  evenly sub-sampled otherwise (fp32). LO / HI
                                  may be -inf / inf (the finite range).
                    linear:LO:HI  evenly spaced values.
                  A second operand is a random permutation of the same sweep.
    eps           the rsqrt(x + eps) guard sweep (see ``eps_sweep``)
    grid          a numerator x denominator grid for division (``div_grid``)

All inputs are drawn in float64 or built from bit patterns, then rounded to the
example's dtype. Only torch is needed.
"""

import math

import torch

from workloads_common import torch_dtype

# Note for reviewers: why this file exists.
#
# Upstream examples/check.py judges every example with one rule:
#     torch.allclose(got, want, atol, rtol)
# That accepts a flushed subnormal, a -0 for +0 and an eps guard that does
# nothing, and it rejects NaN even when both sides are NaN. The tests under
# additional_tests cover exactly those cases, so each suite here adds its own
# rules: bit-exact, class-by-class checks (relu, clamp, leakyrelu, edge), an
# eps-guard check (eps), shape properties (sigtanh) and a ULP measurement (ulp).
# A failure names the class that broke ("zero failed"). The bf16 and fp32
# suites keep upstream's allclose check unmodified.
#
# Purpose of this file:
# -> Builds the special inputs those rules need (every 16-bit pattern,
#    subnormals, points around the eps guard, a division grid), since random
#    data never produces NaN, zeros or subnormals.
# -> workloads_common.make_inputs hands the spec kinds below to sweeps.build.
KINDS = ("bits16", "sweep", "eps", "grid")


def all_values(dtype, shape):
    """Every 16-bit pattern as ``dtype`` (fp16 or bf16)."""
    bits = torch.arange(-32768, 32768, dtype=torch.int16)
    return bits.view(dtype).reshape(shape)


# --------------------------------------------------------------------------
# ULP sweeps
# --------------------------------------------------------------------------


def _index_of(value, fmt):
    bits = torch.tensor([abs(value)], dtype=fmt.dtype).view(fmt.int_dtype).item()
    return -bits if value < 0 else bits


def _values_from_index(idx, fmt):
    sign = 1 << (fmt.width - 1)
    bits = torch.where(idx < 0, idx.abs() | sign, idx)
    bits = torch.where(bits >= sign, bits - (1 << fmt.width), bits)
    return bits.to(fmt.int_dtype).view(fmt.dtype).to(torch.float64)


def _bit_sweep(lo, hi, fmt, n):
    top = torch.finfo(fmt.dtype).max
    lo, hi = max(lo, -top), min(hi, top)
    first, last = _index_of(lo, fmt), _index_of(hi, fmt)
    if last - first + 1 <= n:
        idx = torch.arange(first, last + 1, dtype=torch.int64)
    else:
        idx = torch.linspace(first, last, n, dtype=torch.float64).round().to(torch.int64)
        idx = torch.unique(idx)
    values = _values_from_index(idx, fmt)
    return values[(values >= lo) & (values <= hi)]


def build_values(spec, fmt, numel):
    """``numel`` float64 values for the sweep ``spec`` (see the module docstring)."""
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"empty sweep spec {spec!r}")
    share, extra = divmod(numel, len(tokens))
    parts = []
    for i, token in enumerate(tokens):
        n = share + (extra if i == len(tokens) - 1 else 0)
        kind, *bounds = token.split(":")
        if kind not in ("bits", "linear") or len(bounds) != 2:
            raise ValueError(f"bad sweep token {token!r} in {spec!r}")
        lo, hi = float(bounds[0]), float(bounds[1])
        if kind == "bits":
            values = _bit_sweep(lo, hi, fmt, n)
        else:
            values = torch.linspace(lo, hi, n, dtype=torch.float64)
            values = values.to(fmt.dtype).to(torch.float64)
        if values.numel() == 0:
            raise ValueError(f"sweep token {token!r} selects no values in {fmt.name}")
        reps = -(-n // values.numel())
        parts.append(values.repeat(reps)[:n])
    return torch.cat(parts)


def sweep_inputs(ex, spec, seed):
    from workloads_common import suite_module

    fmt = suite_module("ulp", "ulp_metric").FORMATS[ex.dtype]
    gen = torch.Generator().manual_seed(seed)
    tensors = []
    for k, shape in enumerate(ex.inputs):
        numel = math.prod(shape)
        values = build_values(spec, fmt, numel)
        if k > 0:
            values = values[torch.randperm(numel, generator=gen)]
        tensors.append(values.to(fmt.dtype).reshape(shape))
    return tensors


# --------------------------------------------------------------------------
# rsqrt(x + eps) guard
# --------------------------------------------------------------------------

MANT_BITS = {"fp16": 10, "bf16": 7, "fp32": 23}
N_SUBNORMAL = 1024  # log-spaced points over the subnormal range of the format
N_DOMAIN = 4096  # log-spaced points over the spec domain
DOMAIN_LOG10 = (-7.0, 4.0)  # the spec domain, 1e-7 .. 1e4
NEAR_EPS = (0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 4.0, 10.0, 50.0, 100.0)


def eps_sweep(ex, seed):
    """x for the eps guard: float64, shuffled.

    +0 and -0; ``eps * m`` for m in NEAR_EPS (the points around the guard value);
    log-spaced values over the subnormal range of the format and over the spec
    domain; the rest log-uniform random over the spec domain.
    """
    eps = ex.params["eps"]
    size = math.prod(ex.inputs[0])
    tiny = torch.finfo(torch_dtype(ex)).tiny  # smallest normal
    smallest = tiny * 2.0 ** -MANT_BITS[ex.dtype]  # smallest subnormal
    head = torch.cat(
        [
            torch.tensor([0.0, -0.0], dtype=torch.float64),
            torch.tensor([eps * m for m in NEAR_EPS], dtype=torch.float64),
            torch.logspace(
                math.log10(smallest), math.log10(tiny), N_SUBNORMAL, dtype=torch.float64
            ),
            torch.logspace(*DOMAIN_LOG10, N_DOMAIN, dtype=torch.float64),
        ]
    )
    gen = torch.Generator().manual_seed(seed)
    lo, hi = DOMAIN_LOG10
    exponent = torch.rand(size - len(head), generator=gen, dtype=torch.float64)
    rest = torch.pow(10.0, lo + (hi - lo) * exponent)
    values = torch.cat([head, rest])
    return values[torch.randperm(size, generator=gen)]


def eps_inputs(ex, seed):
    dtype = torch_dtype(ex)
    inputs = [eps_sweep(ex, seed).to(dtype).reshape(ex.inputs[0])]
    if ex.params["form"] == "tensor":
        inputs.append(torch.full(ex.inputs[1], ex.params["eps"], dtype=dtype))
    return inputs


# --------------------------------------------------------------------------
# Division grid
# --------------------------------------------------------------------------

GRID = 256  # the div grid is GRID x GRID = 65536 pairs
N_SUB_NUMERATOR = 16  # extra evenly spaced subnormals on the numerator axis
N_SUB_DENOMINATOR = 96  # ... and on the denominator axis
EXP_BITS = {"fp16": 5, "bf16": 8}


def _positive_specials(fmt):
    """16-bit patterns of the positive values every grid axis must contain."""
    m, e = MANT_BITS[fmt], EXP_BITS[fmt]
    sub_max = (1 << m) - 1  # largest subnormal
    min_norm = 1 << m  # smallest normal
    one = ((1 << (e - 1)) - 1) << m
    two = one + (1 << m)
    three = two + (1 << (m - 1))
    inf = ((1 << e) - 1) << m
    nan = inf | (1 << (m - 1))  # a quiet NaN
    return [
        0, 1, 2, 3, sub_max // 2, sub_max - 1, sub_max,
        min_norm, min_norm + 1, 2 * min_norm, one, two, three,
        inf - 2, inf - 1, inf, nan,
    ]  # fmt: skip


def _fill(chosen, pool, target):
    """Append evenly spaced items of ``pool`` to ``chosen`` until it has ``target``."""
    have = set(chosen)
    pool = [p for p in pool if p not in have]
    need = target - len(chosen)
    step = len(pool) / need
    chosen.extend(pool[int(i * step)] for i in range(need))


def axis_patterns(fmt, n_sub, size=GRID):
    """``size`` distinct 16-bit patterns: the specials of both signs, ``n_sub``
    more evenly spaced subnormals, then evenly spaced finite patterns."""
    m, e = MANT_BITS[fmt], EXP_BITS[fmt]
    inf = ((1 << e) - 1) << m
    sub_max = (1 << m) - 1
    specials = _positive_specials(fmt)
    chosen = list(dict.fromkeys(specials + [p | 0x8000 for p in specials]))
    subnormals = [p for s in (0, 0x8000) for p in range(s + 1, s + sub_max + 1)]
    finite = [p for p in range(0x10000) if (p & 0x7FFF) < inf]
    _fill(chosen, subnormals, len(chosen) + n_sub)
    _fill(chosen, finite, size)
    return chosen


def patterns_to_tensor(patterns, dtype):
    """A tensor of ``dtype`` whose elements have the given 16-bit patterns."""
    bits = torch.tensor(patterns, dtype=torch.int32)
    return torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16).view(dtype)


def div_grid(ex):
    """[a, b]: every numerator pattern paired with every denominator pattern.

    Both axes hold the special values (+-0, +-inf, NaN, the smallest and largest
    subnormal, the smallest normal, +-1, +-2, +-3, the largest finite values)
    plus evenly spaced patterns over the finite range. The denominator axis has
    many more subnormals: the spec asks for "zero or tiny denominators as well
    as different numerator values".
    """
    dtype = torch_dtype(ex)
    numerators = axis_patterns(ex.dtype, N_SUB_NUMERATOR)
    denominators = axis_patterns(ex.dtype, N_SUB_DENOMINATOR)
    a = [p for p in numerators for _ in range(GRID)]  # row-major: a varies slowly
    b = denominators * GRID
    return [patterns_to_tensor(p, dtype).reshape(s) for p, s in zip((a, b), ex.inputs)]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build(ex, seed=0):
    """CPU inputs of example ``ex`` for the sweep kind named by ``ex.init``."""
    kind, _, arg = ex.init.partition("=")
    if kind == "bits16":
        return [all_values(torch_dtype(ex), s) for s in ex.inputs]
    if kind == "sweep":
        return sweep_inputs(ex, arg, seed)
    if kind == "eps":
        return eps_inputs(ex, seed)
    if kind == "grid":
        return div_grid(ex)
    raise ValueError(f"unknown init {ex.init!r}")

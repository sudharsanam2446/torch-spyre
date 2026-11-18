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


"""What the additional spyre-cli suites define: examples, ops and inputs.

A suite is a folder ``additional_tests/<group>/<suite>/`` holding
``<suite>_examples.py``, which defines ``EXAMPLES`` (a dict of :class:`Example`).
Nothing else is needed: building and checking is done by the upstream
``examples/examples.py`` and ``examples/check.py``, which ``run_suites.py``
loads unmodified and points at a suite's registry.

    Example                         one registry entry
    ops()                           the op behind each Example.op
    make_inputs(), torch_dtype()    inputs and dtype of an example
    expected()                      CPU reference result of an example
    suite_names(), load_examples()  find and import a suite
    suite_module()                  import a helper module of one suite
    load_upstream()                 upstream examples.py / check.py, pointed at a suite

Importing this module does not import torch, so tests can be collected without
a Spyre device.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent  # additional_tests/base
ADDITIONAL = BASE.parent  # additional_tests
UPSTREAM = ADDITIONAL.parent.parent / "examples"  # extensions/spyre-cli/examples
DEFAULT_GROUP = "vector-trancendental-math-workloads"

SHAPE16 = (128, 512)  # 65536 elements: every 16-bit pattern once; a multiple of the
#   64-element stick and of 32 cores, so the figures are about the op, not the layout
EXACT = (0.0, 0.0)  # Example.tol of the suites judged by criteria, not allclose

SCALAR_OPS = ("scalar_binary",)  # ops built from a Python scalar (params kind/scalar)
TORCH_DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}


@dataclasses.dataclass(frozen=True)
class Example:
    """One example; the fields upstream's Example has, plus op/workload/title."""

    op: str  # key of ops()
    workload: str  # spec workload id, e.g. "WK-TRN-01"
    title: str  # human-readable operation name
    inputs: tuple  # input shapes, in kernel argument order
    output: tuple  # output shape
    tol: tuple  # (atol, rtol) against the CPU reference
    dtype: str  # fp16, fp32 or bf16
    # How inputs are drawn (see make_inputs):
    #   "randn"          N(0, 1)
    #   "positive"       |N(0, 1)| + 0.5, for ops like div
    #   "uniform:LO:HI"  U(LO, HI), e.g. the spec domain of a workload
    #   "logspace:LO:HI" 10**U(LO, HI), uniform in the exponent
    init: str = "randn"
    # Suite-specific values (a clamp bound, a slope, a ULP limit ...), read only by
    # the suite that sets them.
    params: dict = dataclasses.field(default_factory=dict, compare=False)
    # "" : checked by upstream check.py (allclose against the CPU reference).
    # Otherwise the name of a checker in base/criteria.py (see base/check.py).
    check: str = ""

    @property
    def atol(self):
        return self.tol[0]

    @property
    def rtol(self):
        return self.tol[1]


# --------------------------------------------------------------------------
# Finding suites
# --------------------------------------------------------------------------


def suite_names(group=DEFAULT_GROUP):
    """Names of the suites in ``additional_tests/<group>/``, sorted."""
    folder = ADDITIONAL / group
    if not folder.is_dir():
        return []
    return sorted(
        p.name for p in folder.iterdir() if (p / f"{p.name}_examples.py").is_file()
    )


def suite_dir(suite, group=DEFAULT_GROUP):
    return ADDITIONAL / group / suite


def load_suite(suite, group=DEFAULT_GROUP):
    """The module ``<group>/<suite>/<suite>_examples.py``."""
    path = suite_dir(suite, group) / f"{suite}_examples.py"
    spec = importlib.util.spec_from_file_location(f"{suite}_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_module(suite, name, group=DEFAULT_GROUP):
    """Import ``<group>/<suite>/<name>.py``, a helper module owned by one suite."""
    key = f"{suite}_{name}"
    if key not in sys.modules:
        path = suite_dir(suite, group) / f"{name}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def load_examples(suite, group=DEFAULT_GROUP):
    """The ``EXAMPLES`` dict of a suite."""
    return load_suite(suite, group).EXAMPLES


# --------------------------------------------------------------------------
# Ops and inputs
# --------------------------------------------------------------------------


def ops():
    """The op behind each example, keyed by ``Example.op``."""
    import torch
    import torch.nn.functional as F

    return {
        "exp": torch.exp,
        "rsqrt": torch.rsqrt,
        "reciprocal": torch.reciprocal,
        "div": torch.div,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "gelu": F.gelu,
        "gelu_tanh": lambda a: F.gelu(a, approximate="tanh"),
        "silu": F.silu,
        "swiglu": lambda a, b: F.silu(a) * b,
        "relu": torch.relu,
    }


def build_op(ex):
    """The function compiled for example ``ex``: ``ops()[ex.op]``, or, for the
    ops that carry a constant, one built from ``ex.params``."""
    import torch
    import torch.nn.functional as F

    p = ex.params
    if ex.op == "clamp":
        return lambda t: torch.clamp(t, min=p["lo"], max=p["hi"])
    if ex.op == "leaky_relu":
        return lambda t: F.leaky_relu(t, negative_slope=p["slope"])
    if ex.op == "rsqrt_eps":
        if p["form"] == "tensor":
            return lambda x, e: torch.rsqrt(x + e)
        return lambda x: torch.rsqrt(x + p["eps"])
    if ex.op in SCALAR_OPS:
        binary = {"mul": torch.mul, "div": torch.div, "add": torch.add}[p["kind"]]
        return lambda x: binary(x, p["scalar"])
    return ops()[ex.op]


def symmetric(hi, lo=1e-3):
    """``sweeps`` spec (``init="sweep=..."``): the sweep of [-hi, hi]: every bit pattern with lo <= |x| <= hi, plus an
    even linear sweep of each half.

    A plain bit sweep of [-hi, hi] spends most points near zero (in bf16, 91% of
    [-16, 16] is below 1e-3). Starting the bit sweeps at ``lo`` gives every
    binade proper weight, and the linear halves end at exactly 0, so zero is
    always an input (a zero divisor for ``div``).
    """
    return f"bits:-{hi}:-{lo},bits:{lo}:{hi},linear:-{hi}:0,linear:0:{hi}"


def torch_dtype(ex):
    """The torch dtype of example ``ex``."""
    import torch

    return getattr(torch, TORCH_DTYPES[ex.dtype])


def _draw(init, shape, gen):
    """One float64 CPU tensor of ``shape`` drawn according to ``init``."""
    import torch

    kind, *bounds = init.split(":")
    if kind in ("uniform", "logspace"):
        lo, hi = (float(b) for b in bounds)
        t = lo + (hi - lo) * torch.rand(shape, generator=gen, dtype=torch.float64)
        return 10**t if kind == "logspace" else t
    t = torch.randn(shape, generator=gen, dtype=torch.float64)
    return t.abs() + 0.5 if kind == "positive" else t


def make_inputs(ex, seed=0):
    """CPU inputs for example ``ex``, in its dtype.

    Drawn in float64 and then rounded, so a sweep that reaches into the
    subnormals really contains subnormals of the example's dtype.
    """
    import torch

    # Note for reviewers: the init kinds below need purpose-built inputs, so
    # make_inputs hands them to sweeps.build. Random data would not reliably
    # hit subnormals or exact zeros. Plain kinds (normal, uniform, ...) are
    # drawn here, as upstream does.
    if ex.init.partition("=")[0] in ("bits16", "sweep", "eps", "grid"):
        import sweeps

        return sweeps.build(ex, seed)
    gen = torch.Generator().manual_seed(seed)
    return [_draw(ex.init, shape, gen).to(torch_dtype(ex)) for shape in ex.inputs]


def expected(ex, inputs):
    """CPU result for ``inputs``: computed wide, then rounded to ``ex.dtype``.

    fp16 and bf16 are computed in fp32, as upstream does. fp32 is computed in
    float64, so that an overflow (1/x above the fp32 maximum) becomes inf when
    rounded, as IEEE fp32 division gives.
    """
    import torch

    wide = torch.float64 if ex.dtype == "fp32" else torch.float32
    return build_op(ex)(*[t.to(wide) for t in inputs]).to(torch_dtype(ex))


# --------------------------------------------------------------------------
# Upstream examples.py / check.py, unmodified
# --------------------------------------------------------------------------


def _load(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream(suite, group=DEFAULT_GROUP):
    """Upstream ``examples.py`` and ``check.py``, pointed at ``suite``'s registry.

    Returns ``(examples_module, check_module)``. The upstream files are loaded
    as they are; only the module attributes that name the registry are replaced
    on the loaded module (EXAMPLES, references, make_inputs, expected,
    torch_dtype), all keyed by example name as upstream's are. The result is
    that ``examples_module._compile_one(name, out)`` and
    ``check_module.check_sdk(name, path)`` build and check the suite's examples.

    ``check.py`` binds its names with ``from examples import ...``, so it is
    loaded after the replacement, from ``sys.modules["examples"]``.
    """
    registry = load_examples(suite, group)

    examples = _load("examples", UPSTREAM / "examples.py")
    examples.EXAMPLES = registry
    examples.references = lambda: {n: build_op(ex) for n, ex in registry.items()}
    examples.torch_dtype = lambda name: torch_dtype(registry[name])
    examples.make_inputs = lambda name, seed=0: make_inputs(registry[name], seed)
    examples.expected = lambda name, inputs: expected(registry[name], inputs)
    sys.modules["examples"] = examples

    check = _load("upstream_check", UPSTREAM / "check.py")
    return examples, check
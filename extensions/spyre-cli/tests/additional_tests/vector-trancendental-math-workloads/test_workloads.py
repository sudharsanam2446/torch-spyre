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


"""Tests of the Vector & Transcendental Math suites.

Device tests (skipped when no Spyre device is present): each builds one example
and checks it through the SDK by running ``base/run_all.py <example>``, which
drives the upstream examples.py / check.py (bf16, fp32) or ``base/check.py``
(the suites that set ``Example.check``) in fresh processes, since a device fault
poisons the stream for the rest of the process it happens in.

CPU tests (no device needed):

* registry and wiring: the examples are consistent, and the unmodified upstream
  code sees the allclose suites' registries;
* criteria: with the CPU result standing in for the device every criterion
  passes, and each fault a criterion exists for is caught, by that criterion;
* ``check.check_sdk`` runs end to end against a stand-in for ``spyre_cli.launch``.

    python3 -m pytest test_workloads.py -v
    python3 -m pytest test_workloads.py -v -k bf16            # one suite
    python3 -m pytest test_workloads.py -v -k WK-TRN-03       # one workload
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

GROUP_DIR = Path(__file__).resolve().parent
BASE = GROUP_DIR.parent / "base"
sys.path.insert(0, str(BASE))

from workloads_common import (  # noqa: E402
    UPSTREAM,
    build_op,
    load_examples,
    load_upstream,
    make_inputs,
    ops,
    suite_names,
    torch_dtype,
)

# Purpose of this file:
# -> Each suite folder has one `<suite>_examples.py` that defines an `EXAMPLES`
#    dict. `test_workloads.py` finds those folders, loads each dict and builds
#    tests from it:

GROUP = GROUP_DIR.name
SUITES = {name: load_examples(name, GROUP) for name in suite_names(GROUP)}
ALLCLOSE = {s: e for s, e in SUITES.items() if not any(x.check for x in e.values())}
JUDGED = {s: e for s, e in SUITES.items() if s not in ALLCLOSE}
CASES = [
    pytest.param(s, n, id=f"{s}-{ex.workload}-{n}")
    for s, examples in SUITES.items()
    for n, ex in examples.items()
]
WORKLOADS = {
    "WK-TRN-01", "WK-TRN-02", "WK-TRN-03", "WK-TRN-04",
    "WK-ACT-01", "WK-ACT-02", "WK-ACT-03",
}  # fmt: skip
EXPECTED_SUITES = {
    "bf16", "fp32", "ulp", "sigmoid_tanh", "relu", "clamp", "leakyrelu",
    "rsqrt_eps", "subnormal-zero-nonfinite-edge-cases",
}  # fmt: skip


def test_suites_are_found():
    assert EXPECTED_SUITES <= set(SUITES)


def test_example_names_are_unique_across_suites():
    names = [n for examples in SUITES.values() for n in examples]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("suite", sorted(ALLCLOSE))
def test_allclose_registry_is_consistent(suite):
    assert SUITES[suite]
    for name, ex in SUITES[suite].items():
        assert name.endswith(f"_{suite}")
        assert ex.dtype == suite
        assert ex.workload in WORKLOADS
        assert ex.atol >= 0 and ex.rtol > 0
        assert ex.output and all(ex.inputs)


@pytest.mark.parametrize("suite", sorted(ALLCLOSE))
def test_upstream_code_is_driven_by_the_suite_registry(suite):
    """The unmodified upstream examples.py / check.py see the suite's examples."""
    pytest.importorskip("torch")
    examples, check = load_upstream(suite, GROUP)
    registry = SUITES[suite]
    assert examples.EXAMPLES == registry
    assert examples.__file__ == str(UPSTREAM / "examples.py")
    assert check.__file__ == str(UPSTREAM / "check.py")
    for name, ex in registry.items():
        assert ex.op in ops()
        inputs = examples.make_inputs(name)
        assert [tuple(t.shape) for t in inputs] == [tuple(s) for s in ex.inputs]
        assert all(t.dtype == examples.torch_dtype(name) for t in inputs)
        want = examples.expected(name, inputs)
        assert tuple(want.shape) == tuple(ex.output)
        assert want.dtype == examples.torch_dtype(name)
        same = examples.references()[name](*[t.float() for t in inputs])
        assert same.equal(ops()[ex.op](*[t.float() for t in inputs]))
        assert make_inputs(ex)[0].equal(inputs[0])  # deterministic
        # upstream's own cli_args works on the suite's registry
        assert examples.cli_args(name)[-1] == examples.tensor_spec(ex.output, ex.dtype)


def _raw(t):
    """The bytes of ``t``: torch.equal is False when a NaN is present."""
    return t.contiguous().view(-1).view(torch.uint8)


@pytest.mark.parametrize("suite", sorted(JUDGED))
def test_judged_registry_is_consistent(suite):
    """Every example of a judged suite names a criteria set and builds its inputs."""
    pytest.importorskip("torch")
    import criteria

    for name, ex in SUITES[suite].items():
        assert ex.check in (*criteria.CRITERIA, "ulp"), name
        assert ex.workload in WORKLOADS
        assert ex.dtype in ("fp16", "bf16", "fp32")
        assert ex.output and all(ex.inputs)
        inputs = make_inputs(ex)
        assert [tuple(t.shape) for t in inputs] == [tuple(s) for s in ex.inputs], name
        assert all(t.dtype == torch_dtype(ex) for t in inputs)
        again = make_inputs(ex)
        assert all(_raw(a).equal(_raw(b)) for a, b in zip(again, inputs)), name  # deterministic
        out = build_op(ex)(*inputs)
        assert tuple(out.shape) == tuple(ex.output), name
        if ex.check != "ulp":
            assert criteria.criteria_for(ex), name


# --------------------------------------------------------------------------
# The criteria, on the CPU
# --------------------------------------------------------------------------

JUDGED_CASES = [
    pytest.param(n, ex, id=n) for s, examples in JUDGED.items() for n, ex in examples.items()
]


@pytest.mark.parametrize("name, ex", JUDGED_CASES)
def test_criteria_accept_the_cpu_result(name, ex):
    """With the CPU result as the device result, every example passes (an example
    with no ULP limit is only MEASURED)."""
    import check

    inputs = make_inputs(ex)
    status, detail = check.judge(ex, inputs, build_op(ex)(*inputs))
    assert status in ("PASS", "MEASURED"), detail


def _ex(suite, name):
    return SUITES[suite][name]


def _case(ex):
    """``(inputs, result)``: the example's inputs and the CPU result for them."""
    inputs = make_inputs(ex)
    good = build_op(ex)(*inputs)
    return inputs, good


def _failing(ex, inputs, got):
    import criteria

    return {label for label, c in criteria.criteria_for(ex).items() if c(inputs, got)}


def _set(good, mask, value):
    out = good.clone().reshape(-1)
    out[mask.reshape(-1)] = value
    return out.reshape(good.shape)


def _bump_bits(good, mask, delta):
    """``good`` with the bit pattern of the masked elements moved by ``delta``."""
    out = good.clone().reshape(-1).view(torch.int16)
    m = mask.reshape(-1)
    out[m] = out[m] + delta
    return out.view(good.dtype).reshape(good.shape)


@pytest.mark.parametrize("fmt", ("fp16", "bf16"))
class TestRelu:
    def test_faults(self, fmt):
        ex = _ex("relu", f"relu_{fmt}_exact")
        inputs, good = _case(ex)
        (x,) = inputs
        finite = x.isfinite()
        assert _failing(ex, inputs, _set(good, (x > 0) & (x < 0.09), 0.0)) == {"positive"}
        assert _failing(ex, inputs, _set(good, (x < 0) & finite, -0.0)) == {"negative"}
        assert _failing(ex, inputs, _set(good, x == 0, 0.0)) == {"zero"}
        assert _failing(ex, inputs, _set(good, x == float("-inf"), float("-inf"))) == {"inf/nan"}
        assert _failing(ex, inputs, _set(good, x.isnan(), 0.0)) == {"inf/nan"}


@pytest.mark.parametrize("fmt", ("fp16", "bf16"))
class TestClamp:
    def test_faults(self, fmt):
        ex = _ex("clamp", f"clamp_{fmt}_exact")
        inputs, good = _case(ex)
        (x,) = inputs
        lo, hi = ex.params["lo"], ex.params["hi"]
        one = x == 1.0
        assert _failing(ex, inputs, _bump_bits(good, one, 1)) == {"inside"}
        assert _failing(ex, inputs, _set(good, x == float("-inf"), -2.0)) == {"below"}
        assert _failing(ex, inputs, _set(good, x == float("inf"), 4.0)) == {"above"}
        # off by one at a bound: the bound itself comes back one step away.
        # min <= x <= max is inside, so "inside" fails with it; "edges" must too.
        assert "edges" in _failing(ex, inputs, _bump_bits(good, x == hi, -1))
        assert "edges" in _failing(ex, inputs, _bump_bits(good, x == lo, 1))
        # zero lies inside the bounds, so "inside" sees a changed zero too
        assert "special" in _failing(ex, inputs, _set(good, x == 0, 0.0))
        assert _failing(ex, inputs, _set(good, x.isnan(), lo)) == {"special"}


@pytest.mark.parametrize("name", sorted(JUDGED["leakyrelu"]))
class TestLeakyRelu:
    def test_faults(self, name):
        ex = _ex("leakyrelu", name)
        inputs, good = _case(ex)
        (x,) = inputs
        tiny = torch.finfo(x.dtype).tiny
        assert _failing(ex, inputs, _set(good, (x > 0) & (x < 1), 0.0)) == {"positive"}
        # a negative slope that is one step too large on normal results
        normal_neg = (x < 0) & x.isfinite() & (good.abs() >= tiny)
        assert _failing(ex, inputs, _bump_bits(good, normal_neg, 3)) == {"neg"}
        # subnormal results flushed to zero: only "subnorm" may notice
        flushed = (x < 0) & (good != 0) & good.isfinite() & (good.abs() < tiny)
        assert flushed.any()
        # (the one result at the normal / subnormal boundary can also trip "neg")
        assert "subnorm" in _failing(ex, inputs, _set(good, flushed, 0.0))
        assert _failing(ex, inputs, _set(good, x == 0, 0.0)) == {"zero"}
        assert _failing(ex, inputs, _set(good, x == float("-inf"), 0.0)) == {"inf/nan"}


@pytest.mark.parametrize("name", sorted(JUDGED["sigmoid_tanh"]))
class TestSigmoidTanh:
    def test_faults(self, name):
        ex = _ex("sigmoid_tanh", name)
        inputs, good = _case(ex)
        (x,) = inputs
        inside = (x.abs() <= 16) & x.isfinite()
        # a kink: the output drops by well over one step between 1 and 1.5
        kink = _set(good, (x > 1) & (x < 1.5), (good.double() - 0.05).to(good.dtype)[(x > 1) & (x < 1.5)])
        assert "monotonic" in _failing(ex, inputs, kink)
        assert "bounded" in _failing(ex, inputs, _set(good, x == 2.0, 5.0))
        assert "bounded" in _failing(ex, inputs, _set(good, x == 2.0, float("nan")))
        # one-sided: the positive half is shifted, the negative half is not
        shifted = (good.double() + 0.01).to(good.dtype)
        one_sided = torch.where((x > 0) & inside, shifted, good)
        assert "symmetry" in _failing(ex, inputs, one_sided)


@pytest.mark.parametrize("name", sorted(JUDGED["rsqrt_eps"]))
class TestEpsGuard:
    def test_faults(self, name):
        ex = _ex("rsqrt_eps", name)
        inputs = make_inputs(ex)
        x = inputs[0]
        dtype = x.dtype
        # eps flushed to zero: rsqrt(x + 0) is inf at x = 0 and wrong near zero
        no_guard = torch.rsqrt(x.double()).to(dtype)
        failing = _failing(ex, inputs, no_guard)
        assert "zero" in failing and "subnormal" in failing
        if dtype == torch.float32:
            # rtol 1e-4: a missing guard still moves rsqrt by 5e-4 at x = 1e-3
            assert "small" in failing and "large" in failing
        elif dtype == torch.float16:
            assert "large" not in failing  # far from zero the guard changes nothing
            # The smallest normal fp16 x is 6.1e-5: 1e-6 moves rsqrt by under
            # 0.8%, inside the 2% rtol, so only zero and subnormal x can show a
            # missing guard. In bf16 and fp32 normal x go down to 1e-38.
            assert "small" not in failing
        else:
            assert "large" not in failing and "small" in failing
        # the repo tests' dtype eps instead of 1e-6
        wrong_eps = torch.rsqrt(x.double() + torch.finfo(dtype).eps).to(dtype)
        assert "zero" in _failing(ex, inputs, wrong_eps)
        assert "large" in _failing(ex, inputs, _set(torch.rsqrt(x.double() + 1e-6).to(dtype), x > 1e3, 0.0))


class TestEdgeCases:
    @pytest.mark.parametrize("fmt", ("fp16", "bf16"))
    def test_reciprocal_faults(self, fmt):
        ex = _ex("subnormal-zero-nonfinite-edge-cases", f"reciprocal_{fmt}_edge")
        inputs, good = _case(ex)
        (x,) = inputs
        tiny = torch.finfo(x.dtype).tiny
        sub = (x != 0) & x.isfinite() & (x.abs() < tiny)
        assert _failing(ex, inputs, _set(good, x == 0, 0.0)) == {"zero"}  # 1/0 must be inf
        assert _failing(ex, inputs, _set(good, x.isnan(), 0.0)) == {"inf_nan"}
        assert "subnormal" in _failing(ex, inputs, _set(good, sub, 0.0))
        assert "normal" in _failing(ex, inputs, _bump_bits(good, x == 1.0, 3))

    @pytest.mark.parametrize("fmt", ("fp16", "bf16"))
    def test_div_faults(self, fmt):
        ex = _ex("subnormal-zero-nonfinite-edge-cases", f"div_{fmt}_edge")
        inputs, good = _case(ex)
        a, b = inputs
        tiny = torch.finfo(b.dtype).tiny
        assert "zero_divisor" in _failing(ex, inputs, _set(good, (b == 0) & (a == 1.0), 0.0))
        assert "non_finite" in _failing(ex, inputs, _set(good, b.isinf() & (a == 1.0), 1.0))
        tiny_b = (b != 0) & b.isfinite() & (b.abs() < tiny) & a.isfinite() & (a != 0)
        assert "tiny_divisor" in _failing(ex, inputs, _set(good, tiny_b, 0.0))

    @pytest.mark.parametrize("name", sorted(n for n in JUDGED["subnormal-zero-nonfinite-edge-cases"] if n.startswith("scalar_")))
    def test_scalar_ops_accept_either_reference(self, name):
        """The scalar rounded to the dtype is also an acceptable device result."""
        import criteria

        ex = _ex("subnormal-zero-nonfinite-edge-cases", name)
        inputs = make_inputs(ex)
        for reference in criteria.edge_references(ex):
            got = reference(*inputs)
            assert not _failing(ex, inputs, got), name


@pytest.mark.parametrize("name", ("exp_fp16_ulp", "exp_bf16_ulp", "relu_fp16_ulp", "tanh_fp16_ulp"))
def test_ulp_verdicts(name, monkeypatch):
    import check

    monkeypatch.delenv("SPYRE_MAX_ULP", raising=False)
    ex = _ex("ulp", name)
    inputs, good = _case(ex)
    status, _ = check.judge(ex, inputs, good)
    assert status == ("MEASURED" if ex.params["limit"] is None else "PASS")

    positive = good.isfinite() & (good > 0) & (good < 1000)
    three_ulp = _bump_bits(good, positive, 3)
    status, detail = check.judge(ex, inputs, three_ulp)
    assert status == ("MEASURED" if ex.params["limit"] is None else "FAIL"), detail
    # --max-ulp style override gates every example
    monkeypatch.setenv("SPYRE_MAX_ULP", "1.5")
    assert check.judge(ex, inputs, three_ulp)[0] == "FAIL"
    assert check.judge(ex, inputs, good)[0] == "PASS"
    # NaN where the reference is finite is a class mismatch: a FAIL at any limit
    nan_result = _set(good, good.isfinite() & (good != 0), float("nan"))
    assert check.judge(ex, inputs, nan_result)[0] == "FAIL"


# --------------------------------------------------------------------------
# check.check_sdk against a stand-in for spyre_cli.launch
# --------------------------------------------------------------------------


@pytest.fixture
def stub_device(monkeypatch):
    """Make ``.to("spyre")`` a no-op and ``spyre_cli.launch`` run the example on the CPU."""
    real_to, real_empty = torch.Tensor.to, torch.empty

    def to(self, *args, **kwargs):
        if args and args[0] == "spyre":
            return self
        return real_to(self, *args, **kwargs)

    def empty(*args, **kwargs):
        if kwargs.get("device") == "spyre":
            kwargs.pop("device")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "empty", empty)

    state = types.SimpleNamespace(seen=None, fault=None, ex=None, raises=None)

    def launch(*args, path=None):
        *tensors, out = args
        state.seen = len(tensors)
        if state.raises:
            raise state.raises
        ex = state.ex
        if ex.op == "rsqrt_eps" and ex.params["form"] == "scalar":
            x, const = tensors  # the bundle takes (x, constant, out)
            result = torch.rsqrt(x + const)
        else:
            result = build_op(ex)(*tensors)
        out.copy_(state.fault(ex, result) if state.fault else result)
        return object()

    monkeypatch.setitem(sys.modules, "spyre_cli", types.SimpleNamespace(launch=launch))
    return state


@pytest.mark.parametrize(
    "suite, name",
    [
        ("relu", "relu_fp16_exact"),
        ("clamp", "clamp_bf16_exact"),
        ("sigmoid_tanh", "tanh_fp16_props"),
        ("rsqrt_eps", "rsqrt_eps_scalar_fp16"),
        ("rsqrt_eps", "rsqrt_eps_tensor_fp32"),
        ("subnormal-zero-nonfinite-edge-cases", "div_bf16_edge"),
        ("ulp", "exp_fp16_ulp"),
    ],
)
def test_check_sdk_end_to_end(suite, name, stub_device, tmp_path):
    import check

    ex = _ex(suite, name)
    stub_device.ex = ex
    status, detail = check.check_sdk(ex, name, tmp_path)
    assert status in ("PASS", "MEASURED"), detail
    # the eps guard's scalar form passes the constant as an extra 0-d argument
    assert stub_device.seen == len(ex.inputs) + (1 if name == "rsqrt_eps_scalar_fp16" else 0)

    # a fault in the "device" result is reported as a FAIL
    stub_device.fault = lambda ex, r: _set(r, r.isfinite() & (r != 0), 0.0)
    status, _ = check.check_sdk(ex, name, tmp_path)
    assert status == "FAIL"


def test_check_sdk_reports_a_launch_exception(stub_device, tmp_path):
    import check

    ex = _ex("clamp", "clamp_fp16_exact")
    stub_device.ex = ex
    stub_device.raises = RuntimeError("argument count mismatch")
    status, detail = check.check_sdk(ex, "clamp_fp16_exact", tmp_path)
    assert status == "FAIL"
    assert "launch raised RuntimeError" in detail and "argument count" in detail


def test_run_all_status_line_is_parsed():
    import run_all

    assert run_all._status("x_ulp sdk: MEASURED max_ulp=1.2") == ("MEASURED", "max_ulp=1.2")
    assert run_all._status("x sdk: PASS a=PASS") == ("PASS", "a=PASS")
    assert run_all._status("nothing here") == ("FAIL", "no result")
    assert run_all.passed({"compile": "PASS", "sdk": "MEASURED"})
    assert not run_all.passed({"compile": "PASS", "sdk": "FAIL"})
    assert not run_all.passed({"compile": "FAIL", "sdk": "SKIP"})


# --------------------------------------------------------------------------
# Device tests
# --------------------------------------------------------------------------


def _spyre_available():
    probe = "import torch; print(torch.accelerator.current_accelerator())"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "spyre" in result.stdout


needs_device = pytest.mark.skipif(not _spyre_available(), reason="needs a Spyre device")


@needs_device
@pytest.mark.parametrize("suite, name", CASES)
def test_example(suite, name, tmp_path):
    cmd = [sys.executable, str(BASE / "run_all.py"), "--group", GROUP]
    cmd += ["--out", str(tmp_path), name]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

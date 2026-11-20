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


"""Build and check for the examples that upstream's ``allclose`` cannot judge.

Suites whose examples set ``Example.check`` (ulp, relu, clamp, ...) are built
and checked here instead of by upstream ``examples.py`` / ``check.py``, because:

* their inputs hold NaN, inf, zeros and subnormals on purpose, and upstream's
  ``_compile_one`` compares with ``torch.allclose``, which rejects a NaN;
* the verdict is a set of criteria (bit-exact, monotonic, ULP ...), not one
  tolerance (see ``criteria.py``).

Two functions, mirroring upstream's pair and the line ``run_all.py`` parses:

    compile_one(ex, name, out)   torch.compile the op on the device, run it
                                 once, copy the one kernel it added to out/name
    check_sdk(ex, name, path)    spyre_cli.launch over the example's inputs and
                                 apply the criteria; returns (status, detail)

``status`` is PASS or FAIL, or MEASURED for a ULP example without a limit (the
figures are recorded, there is nothing to pass). A ULP limit can be forced for
every example with the environment variable ``SPYRE_MAX_ULP``.

An exception raised while launching is reported on the result line (type and
message) instead of being lost.
"""

import glob
import os
import shutil
import sys
from pathlib import Path

from workloads_common import build_op, make_inputs, suite_module, torch_dtype

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
# -> Builds each example with torch.compile, launches the saved kernel with
#    spyre_cli.launch, and judges the result.
# -> It is used only by the suites that need stricter rules than allclose.


def _kernel_dirs(cache):
    return set(glob.glob(os.path.join(cache, "inductor-spyre", "*", "spyreCodeDir")))


def compile_one(ex, name, out):
    """Compile ``ex`` into ``out/name``. Runs in a child process whose
    TORCHINDUCTOR_CACHE_DIR is already set. Compiling does not compare numerics:
    a kernel whose results are wrong is exactly the one worth measuring."""
    import torch

    cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    fn = torch.compile(build_op(ex), backend="inductor")
    on_device = [t.to("spyre") for t in make_inputs(ex)]

    # Only count kernels that torch.compile adds for the op: anything the
    # host-to-device copies compiled beforehand is not the kernel under test.
    before = _kernel_dirs(cache)
    fn(*on_device).cpu()  # compile and run once
    kernels = sorted(_kernel_dirs(cache) - before)
    if len(kernels) != 1:
        sys.exit(f"{name}: expected 1 kernel, torch.compile produced {len(kernels)}")

    dest = Path(out) / name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(kernels[0]), dest)
    print(f"{name}: wrote {dest}")


def launch_args(ex):
    """Extra device arguments the bundle takes between the inputs and the output.

    A Python scalar in the op, such as ``rsqrt(x + 1e-6)`` or ``x * 6e-8``, is
    compiled as a 0-d device tensor (``spyre_constant_tensor``), so the bundle
    takes ``(x, constant, out)``. A standalone launch must pass it too, or
    ``launch`` fails with "Number of inputs provided (2) does not match number
    of inputs expected (3)".

    The ``scalar_binary`` constant is built by ``spyre_constant_tensor`` itself,
    as the compiled graph does, so a NaN, inf or subnormal constant reaches the
    kernel the same way and is not altered by a host-to-device copy first.
    """
    import torch

    if ex.op == "rsqrt_eps" and ex.params["form"] == "scalar":
        return [torch.tensor(ex.params["eps"], dtype=torch_dtype(ex)).to("spyre")]
    if ex.op == "scalar_binary":
        from torch_spyre._inductor.op_spec import spyre_constant_tensor

        return [
            spyre_constant_tensor(
                ex.params["scalar"], torch.device("spyre"), torch_dtype(ex)
            )
        ]
    return []


def _ulp(ex, inputs, got):
    ulp_metric = suite_module("ulp", "ulp_metric")

    report = ulp_metric.measure(
        inputs, got, ulp_metric.reference(ex.op, inputs), ex.dtype
    )
    limit = os.environ.get("SPYRE_MAX_ULP")
    limit = float(limit) if limit else ex.params.get("limit")
    status = ulp_metric.verdict(report, limit)
    limit_text = "none" if limit is None else limit
    return status, f"{ulp_metric.one_line(report)} limit={limit_text}"


def judge(ex, inputs, got):
    """``(status, detail)`` for the device result ``got`` of ``inputs``."""
    if ex.check == "ulp":
        return _ulp(ex, inputs, got)
    from criteria import evaluate

    ok, detail = evaluate(ex, inputs, got)
    return ("PASS" if ok else "FAIL"), detail


def check_sdk(ex, name, path):
    import spyre_cli
    import torch

    inputs = make_inputs(ex)
    out = torch.empty(ex.output, dtype=torch_dtype(ex), device="spyre")
    args = [t.to("spyre") for t in inputs] + launch_args(ex)
    try:
        runner = spyre_cli.launch(*args, out, path=path)
        got = out.cpu()
        del runner
    except Exception as exc:  # report it; the table would show "no result"
        return "FAIL", f"launch raised {type(exc).__name__}: {str(exc)[:500]}"
    return judge(ex, inputs, got)

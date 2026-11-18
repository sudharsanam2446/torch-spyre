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


"""Run the additional spyre-cli suites one by one, through the SDK.

Usage: python run_all.py [--group G] [--out DIR] [suite | example ...]

With no arguments every suite of the group (bf16, fp32, ...) is run, in order.
Name suites to run only those, or examples (names are unique across suites) to
run only those.

Nothing is reimplemented here. For each example this script runs, in a fresh
process, the upstream code in ``extensions/spyre-cli/examples``, unmodified
(examples that set ``Example.check``, the suites whose verdict is not one
``allclose``, are built and judged by ``base/check.py`` instead):

1. build: ``examples.py``'s ``_compile_one``  (torch.compile on the device, one
   kernel, copied to <DIR>/<suite>/<example>; default <suite>/build/<example>)
2. check: ``check.py``'s ``check_sdk``  (spyre_cli.launch on random inputs, must
   match the CPU reference within the example's (atol, rtol))

``spyre launch`` (upstream's cli check) is not run. The suite's registry is
swapped in by ``workloads_common.load_upstream``. A fresh process per example
matters: a device fault leaves the stream in an error state for the rest of the
process it happens in. A failing example does not stop the run; if its build
fails there is no bundle, so its check is reported as SKIP. A summary table is
printed at the end, and the exit status is 0 only if every selected example
builds and passes.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workloads_common import (  # noqa: E402
    DEFAULT_GROUP,
    load_examples,
    load_upstream,
    suite_dir,
    suite_names,
)

SELF = Path(__file__).resolve()
OK = ("PASS", "MEASURED")  # MEASURED: a ULP figure with no limit to pass or fail


# --------------------------------------------------------------------------
# Child processes: one build or one check, via the upstream code
# --------------------------------------------------------------------------


def _child(mode, group, suite, name, target):
    ex = load_examples(suite, group)[name]
    if ex.check:  # judged by base/check.py (criteria, ULP ...), not by allclose
        import check as custom

        if mode == "build":
            custom.compile_one(ex, name, target)  # exits non-zero on failure
            return
        status, detail = custom.check_sdk(ex, name, Path(target))
        print(f"{name} sdk: {status} {detail}")
        sys.exit(0 if status in OK else 1)

    examples, check = load_upstream(suite, group)
    if mode == "build":
        examples._compile_one(name, target)  # exits non-zero on failure
        return
    ok, detail = check.check_sdk(name, Path(target))
    print(f"{name} sdk: {'PASS' if ok else 'FAIL'} {detail}")
    sys.exit(0 if ok else 1)


# --------------------------------------------------------------------------
# Parent: run every selected example, then summarise
# --------------------------------------------------------------------------


def _run(cmd, timeout, env=None):
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
    )
    return result.returncode, result.stdout + result.stderr


def _status(output):
    """PASS/FAIL (and detail) of the ``<name> sdk:`` line in check output."""
    match = re.search(r"^\S+ sdk: (PASS|FAIL|MEASURED) ?(.*)$", output, re.MULTILINE)
    return (match.group(1), match.group(2)) if match else ("FAIL", "no result")


def _reason(name, output):
    """The most informative line of a failed build's output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    own = [line for line in lines if name in line and not line.startswith("failed:")]
    errors = [line for line in lines if re.match(r"\w*(Error|Exception)\b", line)]
    return (own or errors or lines or ["compile failed"])[-1]


def run_one(group, suite, name, out):
    """Build and check ``name``; return its result row."""
    row = {"compile": "FAIL", "sdk": "SKIP", "detail": ""}
    with tempfile.TemporaryDirectory() as cache:
        # A private Inductor cache, so exactly one kernel directory is produced.
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache)
        cmd = [sys.executable, str(SELF), "--child", "build", group, suite, name, out]
        try:
            code, output = _run(cmd, 1800, env)
        except subprocess.TimeoutExpired:
            row["detail"] = "compile timed out"
            return row
    if code != 0:
        row["detail"] = _reason(name, output)
        return row
    row["compile"] = "PASS"

    cmd = [
        sys.executable, str(SELF), "--child", "check", group, suite, name,
        str(Path(out) / name),
    ]  # fmt: skip
    try:
        _, output = _run(cmd, 1200)
    except subprocess.TimeoutExpired:
        row["sdk"] = "FAIL"
        row["detail"] = "check timed out"
        return row
    row["sdk"], row["detail"] = _status(output)
    return row


def passed(row):
    return row["compile"] == "PASS" and row["sdk"] in OK


def select(group, targets):
    """``[(suite, example name)]`` for the suite and example names in ``targets``."""
    suites = {s: load_examples(s, group) for s in suite_names(group)}
    if not targets:
        targets = list(suites)
    chosen, unknown = [], []
    for target in targets:
        if target in suites:
            chosen += [(target, n) for n in suites[target]]
            continue
        owners = [s for s, examples in suites.items() if target in examples]
        if owners:
            chosen.append((owners[0], target))
        else:
            unknown.append(target)
    return suites, chosen, unknown


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(*sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", metavar="suite|example")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--out", help="kernel directory (default: <suite>/build)")
    args = parser.parse_args()

    suites, chosen, unknown = select(args.group, args.targets)
    if unknown:
        parser.error(
            f"unknown suite or example: {' '.join(unknown)} "
            f"(suites: {' '.join(suites)})"
        )

    rows = {}
    for suite, name in chosen:
        out = str(Path(args.out) / suite if args.out else suite_dir(suite, args.group) / "build")
        print(f"== {suite}/{name}", flush=True)
        rows[name] = run_one(args.group, suite, name, out)
        print(f"   {'PASS' if passed(rows[name]) else 'FAIL'}", flush=True)

    print()
    header = (
        f"{'Suite':<12} {'Workload':<10} {'Example':<32} {'Operation':<26} "
        f"{'compile':<8} {'sdk':<9} detail"
    )
    print(header)
    print("-" * len(header))
    for suite, name in chosen:
        ex, row = suites[suite][name], rows[name]
        print(
            f"{suite:<12} {ex.workload:<10} {name:<32} {ex.title:<26} "
            f"{row['compile']:<8} {row['sdk']:<9} {row['detail']}"
        )
    ok = sum(passed(row) for row in rows.values())
    print(f"\n{ok}/{len(chosen)} examples passed")
    failed = [n for _, n in chosen if not passed(rows[n])]
    if failed:
        print("failed: " + " ".join(failed))
    sys.exit(0 if ok == len(chosen) else 1)


if __name__ == "__main__":
    main()

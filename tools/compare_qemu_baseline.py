#!/usr/bin/env python3
"""Compare current tree vs baseline with host parity and QEMU relative timing.

QEMU is used here for:
- functional testing of the Cortex-M55 + MVE path
- relative-only timing against a previous baseline commit

QEMU wall time is not treated as a real M55 performance number.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


CHECKSUM_RE = re.compile(r"checksum:\s*(0x[0-9a-fA-F]+)")
MVE_RE = re.compile(
    r"mve-build:\s*compiler_mve=(\d+)\s+cmsis_mvei=(\d+)\s+autovec_disabled=(\d+)\s+probe=(0x[0-9a-fA-F]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-commit", default="1d2347c")
    parser.add_argument("--input-seq-len", type=int, default=128)
    parser.add_argument("--ffn-dim", type=int, default=48)
    parser.add_argument("--runs", type=int, default=3, help="Number of QEMU timing runs per build")
    parser.add_argument("--keep-worktree", action="store_true", help="Keep the extracted baseline source tree in /tmp")
    return parser.parse_args()


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def export_git_snapshot(repo_root: Path, commit: str, out_dir: Path) -> None:
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=repo_root,
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        import tarfile

        with tarfile.open(fileobj=archive.stdout, mode="r|") as tf:
            tf.extractall(path=out_dir)
    finally:
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError(f"git archive failed for {commit}")


def extract_checksum(output: str) -> int:
    match = CHECKSUM_RE.search(output)
    if match is None:
        raise RuntimeError("checksum line not found")
    return int(match.group(1), 16)


def extract_mve(output: str) -> tuple[int, int, int, str] | None:
    match = MVE_RE.search(output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)


def locate_cmsis5(repo_root: Path) -> Path:
    env_path = os.environ.get("CMSIS_5_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            repo_root.parent / "jax_to_embedded" / "cm4-qemu" / "CMSIS_5",
            repo_root.parent / "CMSIS_5",
        ]
    )
    for candidate in candidates:
        if (candidate / "CMSIS" / "Core" / "Include" / "core_cm55.h").is_file():
            return candidate
    raise FileNotFoundError("unable to locate CMSIS_5 checkout with core_cm55.h")


def ensure_tmp_cmsis_dsp_link(repo_root: Path) -> None:
    target = repo_root.parent / "CMSIS-DSP"
    if not target.is_dir():
        raise FileNotFoundError(f"CMSIS-DSP checkout not found: {target}")

    link = Path("/tmp/CMSIS-DSP")
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError("/tmp/CMSIS-DSP exists and is not the expected symlink")
        link.unlink()
    link.symlink_to(target)


def configure_and_build(
    source_dir: Path,
    build_dir: Path,
    repo_root: Path,
    cmsis5_dir: Path,
    input_seq_len: int,
    ffn_dim: int,
    qemu: bool,
) -> None:
    cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DMODEL_INPUT_SEQ_LEN={input_seq_len}",
        f"-DMODEL_FFN_DIM={ffn_dim}",
    ]
    if qemu:
        cmd.extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={repo_root / 'cmake' / 'toolchains' / 'arm-none-eabi-gcc.cmake'}",
                "-DARM_M55_QEMU=ON",
                f"-DCMSIS_5_DIR={cmsis5_dir}",
            ]
        )
    run_checked(cmd, cwd=source_dir)
    run_checked(["cmake", "--build", str(build_dir), "-j4"], cwd=source_dir)


def run_binary(binary: Path, cwd: Path) -> str:
    return run_checked([str(binary)], cwd=cwd)


def run_qemu(binary: Path, cwd: Path, qemu_system_arm: str) -> tuple[str, float]:
    cmd = [
        qemu_system_arm,
        "-machine",
        "mps3-an547",
        "-cpu",
        "cortex-m55",
        "-nographic",
        "-serial",
        "null",
        "-monitor",
        "null",
        "-semihosting-config",
        "enable=on,target=native",
        "-kernel",
        str(binary),
    ]
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return completed.stdout + completed.stderr, elapsed


def summarize_times(times: list[float]) -> tuple[float, float]:
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.pstdev(times)


def build_and_test_variant(
    label: str,
    source_dir: Path,
    repo_root: Path,
    cmsis5_dir: Path,
    input_seq_len: int,
    ffn_dim: int,
    runs: int,
    qemu_system_arm: str,
) -> dict[str, object]:
    build_root = Path(tempfile.mkdtemp(prefix=f"arm-mve-{label}-build-", dir="/tmp"))
    host_build = build_root / "host"
    qemu_build = build_root / "qemu"

    configure_and_build(source_dir, host_build, repo_root, cmsis5_dir, input_seq_len, ffn_dim, qemu=False)
    configure_and_build(source_dir, qemu_build, repo_root, cmsis5_dir, input_seq_len, ffn_dim, qemu=True)

    host_output = run_binary(host_build / "tiled_attention", source_dir)
    qemu_times: list[float] = []
    qemu_output = ""
    for _ in range(runs):
        qemu_output, elapsed = run_qemu(qemu_build / "tiled_attention", source_dir, qemu_system_arm)
        qemu_times.append(elapsed)

    host_checksum = extract_checksum(host_output)
    qemu_checksum = extract_checksum(qemu_output)
    mve = extract_mve(qemu_output)
    if host_checksum != qemu_checksum:
        raise RuntimeError(f"{label}: host checksum 0x{host_checksum:08x} != qemu checksum 0x{qemu_checksum:08x}")
    if label == "current" and (mve is None):
        raise RuntimeError("current: qemu output is missing the mve-build verification banner")
    if mve is not None:
        compiler_mve, cmsis_mvei, autovec_disabled, probe = mve
    else:
        compiler_mve, cmsis_mvei, autovec_disabled, probe = None, None, None, "n/a"
    if label == "current" and (compiler_mve != 1 or cmsis_mvei != 1):
        raise RuntimeError(f"{label}: QEMU build is not on the expected MVE path")

    median_s, stdev_s = summarize_times(qemu_times)
    return {
        "label": label,
        "checksum": host_checksum,
        "compiler_mve": compiler_mve,
        "cmsis_mvei": cmsis_mvei,
        "mve_probe": probe,
        "autovec_disabled": autovec_disabled,
        "median_s": median_s,
        "stdev_s": stdev_s,
        "runs": qemu_times,
    }


def print_result(result: dict[str, object]) -> None:
    compiler_mve = result["compiler_mve"]
    cmsis_mvei = result["cmsis_mvei"]
    print(
        f"{result['label']}: checksum=0x{result['checksum']:08x} "
        f"compiler_mve={compiler_mve if compiler_mve is not None else 'unknown'} "
        f"cmsis_mvei={cmsis_mvei if cmsis_mvei is not None else 'unknown'} "
        f"probe={result['mve_probe']}"
    )
    print(
        f"{result['label']}: qemu-relative median={result['median_s']:.3f}s "
        f"stdev={result['stdev_s']:.3f}s runs={len(result['runs'])}"
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    qemu_system_arm = shutil.which("qemu-system-arm")
    if qemu_system_arm is None:
        raise FileNotFoundError("qemu-system-arm not found in PATH")

    cmsis5_dir = locate_cmsis5(repo_root)
    ensure_tmp_cmsis_dsp_link(repo_root)

    baseline_dir = Path(tempfile.mkdtemp(prefix="arm-mve-baseline-", dir="/tmp"))

    try:
        export_git_snapshot(repo_root, args.baseline_commit, baseline_dir)

        current = build_and_test_variant(
            "current",
            repo_root,
            repo_root,
            cmsis5_dir,
            args.input_seq_len,
            args.ffn_dim,
            args.runs,
            qemu_system_arm,
        )
        baseline = build_and_test_variant(
            "baseline",
            baseline_dir,
            repo_root,
            cmsis5_dir,
            args.input_seq_len,
            args.ffn_dim,
            args.runs,
            qemu_system_arm,
        )

        print_result(current)
        print_result(baseline)

        speedup = baseline["median_s"] / current["median_s"]
        print(
            f"relative-only qemu speedup: {speedup:.3f}x "
            f"(baseline/current, input_seq_len={args.input_seq_len})"
        )
        print("note: qemu timing is only for relative comparison between these two binaries, not real M55 time")
        return 0
    finally:
        try:
            if not args.keep_worktree:
                shutil.rmtree(baseline_dir, ignore_errors=True)
        finally:
            if args.keep_worktree:
                print(f"kept baseline source tree at {baseline_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare the host C binary against the PyTorch fixed-point reference."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


CHECKSUM_RE = re.compile(r"checksum:\s*(0x[0-9a-fA-F]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build-host-1200-nopool/tiled_attention"),
        help="Host C binary to execute",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter with torch installed",
    )
    parser.add_argument(
        "--torch-script",
        type=Path,
        default=Path("tools/torch_reference.py"),
        help="PyTorch reference script",
    )
    parser.add_argument("--input-seq-len", type=int, default=1200)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--max-abs-diff",
        type=int,
        default=0,
        help="Allowed absolute q7 difference between the two outputs",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def extract_checksum(output: str) -> int:
    match = CHECKSUM_RE.search(output)
    if match is None:
        raise RuntimeError("checksum line not found")
    return int(match.group(1), 16)


def load_q7_bytes(path: Path) -> list[int]:
    raw = path.read_bytes()
    return [value if value < 128 else value - 256 for value in raw]


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"host binary not found: {args.binary}")
    if not args.python.is_file():
        raise FileNotFoundError(f"python interpreter not found: {args.python}")
    if not args.torch_script.is_file():
        raise FileNotFoundError(f"torch script not found: {args.torch_script}")

    expected_elems = args.input_seq_len * args.num_classes

    c_fd, c_dump_name = tempfile.mkstemp(prefix="torch-parity-c-", suffix=".q7", dir="/tmp")
    py_fd, py_dump_name = tempfile.mkstemp(prefix="torch-parity-py-", suffix=".q7", dir="/tmp")
    os.close(c_fd)
    os.close(py_fd)
    c_dump = Path(c_dump_name)
    py_dump = Path(py_dump_name)

    try:
        c_env = os.environ.copy()
        c_env["TILED_ATTENTION_DUMP"] = str(c_dump)
        c_stdout = run_checked([str(args.binary)], env=c_env)
        py_stdout = run_checked(
            [
                str(args.python),
                str(args.torch_script),
                "--input-seq-len",
                str(args.input_seq_len),
                "--dump-output",
                str(py_dump),
            ]
        )

        c_values = load_q7_bytes(c_dump)
        py_values = load_q7_bytes(py_dump)
    finally:
        c_dump.unlink(missing_ok=True)
        py_dump.unlink(missing_ok=True)

    if len(c_values) != expected_elems:
        raise RuntimeError(f"C dump has {len(c_values)} values, expected {expected_elems}")
    if len(py_values) != expected_elems:
        raise RuntimeError(f"Torch dump has {len(py_values)} values, expected {expected_elems}")

    abs_diffs = [abs(a - b) for a, b in zip(c_values, py_values)]
    max_abs_diff = max(abs_diffs, default=0)
    mismatch_count = sum(diff > args.max_abs_diff for diff in abs_diffs)
    c_checksum = extract_checksum(c_stdout)
    py_checksum = extract_checksum(py_stdout)

    print(f"C checksum:     0x{c_checksum:08x}")
    print(f"Torch checksum: 0x{py_checksum:08x}")
    print(f"Max abs diff:   {max_abs_diff}")
    print(f"Mismatches:     {mismatch_count}")

    if mismatch_count != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

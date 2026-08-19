#!/usr/bin/env python3
"""Fail when generated protocol types differ from the checked-in file."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from generate_types import DEFAULT_OUTPUT, DEFAULT_SCHEMA, generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected = generate(args.schema)
    try:
        actual = args.output.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"missing generated file: {args.output}", file=sys.stderr)
        return 1

    if actual == expected:
        print("generated protocol types are current")
        return 0

    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(args.output),
        tofile="generated output",
    )
    sys.stderr.writelines(diff)
    print("run: python schema/generate_types.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

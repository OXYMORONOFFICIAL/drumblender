#!/usr/bin/env python3
"""Compatibility wrapper around scripts/postprocess_results.py."""

from __future__ import annotations

import argparse
from typing import List, Optional

from postprocess_results import main as postprocess_main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=str, help="Run directory or exported bundle directory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report-dir", default=None, help="Directory for generated reports")
    args = parser.parse_args(argv)

    forwarded = [args.input_path]
    if args.report_dir is not None:
        forwarded += ["--report-dir", args.report_dir]
    return postprocess_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

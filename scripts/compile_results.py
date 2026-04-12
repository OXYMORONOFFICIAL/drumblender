#!/usr/bin/env python3
"""Compatibility wrapper around scripts/postprocess_results.py."""

from __future__ import annotations

import argparse
from typing import List, Optional

from postprocess_results import main as postprocess_main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indir", type=str, help="Run directory or exported bundle directory")
    parser.add_argument("type", nargs="?", default="all")
    parser.add_argument("--input-format", default="auto")
    parser.add_argument("--run-name-regex", default=None)
    parser.add_argument("-o", "--outfile", default=None, help="Path to summary TeX output")
    parser.add_argument("--report-dir", default=None, help="Directory for generated reports")
    args = parser.parse_args(argv)

    forwarded = [args.indir]
    if args.report_dir is not None:
        forwarded += ["--report-dir", args.report_dir]
    if args.outfile is not None:
        forwarded += ["--tex-out", args.outfile]
    return postprocess_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

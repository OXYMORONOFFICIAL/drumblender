#!/usr/bin/env python3
import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "recon" / "control_live.py"
    runpy.run_path(str(target), run_name="__main__")

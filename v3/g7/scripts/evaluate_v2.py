#!/usr/bin/env python3
"""Fresh G7 batch evaluation using the unified output contract."""
from pathlib import Path
import sys
V3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3_ROOT / "scripts"))
from run_all_eval import main
if __name__ == "__main__":
    raise SystemExit(main(["--groups", "g7", *sys.argv[1:]]))

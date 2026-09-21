#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from refined_evaluators.batch import run
if __name__ == '__main__':sys.exit(run('P43',Path(__file__).resolve().parents[1]))

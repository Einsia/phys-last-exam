#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str((Path(__file__).resolve().parent / "_shared")))
from refined_evaluators.runtime import run
from observed_support import install
install()
if __name__ == '__main__':sys.exit(run('P43'))

#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str((Path(__file__).resolve().parent / "_shared")))
from refined_evaluators.runtime import run
from refined_evaluators.tasks import TASKS
from measured_center import p38
TASKS['P38']=p38
if __name__ == '__main__':
    from observed_zero import install
    install()
    sys.exit(run('P38'))

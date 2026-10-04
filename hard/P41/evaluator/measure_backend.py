#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str((Path(__file__).resolve().parent / "_shared")))
from refined_evaluators.runtime import run
from refined_evaluators.tasks import TASKS
from measured_discharge import p41
TASKS['P41']=p41
if __name__ == '__main__':sys.exit(run('P41'))

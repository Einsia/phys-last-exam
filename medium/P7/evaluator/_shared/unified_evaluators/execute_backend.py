"""Bound OpenCV worker counts before executing an unchanged measurement program."""
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('MPLCONFIGDIR','/tmp/g1_g9_matplotlib')
os.environ.setdefault('MPLBACKEND','Agg')
if os.environ.get('EVALUATOR_USE_OPENCV4') == '1':
    dependency = Path(__file__).resolve().parents[1]/'.runtime/opencv4'
    if dependency.exists():
        sys.path.insert(0,str(dependency))
import cv2
cv2.setNumThreads(int(os.environ.get('EVALUATOR_CV_THREADS','2')))

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')

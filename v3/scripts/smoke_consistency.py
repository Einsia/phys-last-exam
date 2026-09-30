#!/usr/bin/env python3
"""Run the two user-supplied P19 examples through the real V3 gate and physics."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unified_evaluators.consistency import add_arguments, Settings
from unified_evaluators.runtime import main as evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(os.environ.get('VDMBENCH_DATA_ROOT', Path.cwd())))
    parser.add_argument('--output-root', type=Path, default=ROOT/'work/consistency_smoke')
    add_arguments(parser)
    args = parser.parse_args()
    run = (args.output_root / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    run.mkdir(parents=True)
    source = args.source_root
    reference = source/'g2/P19/first_frames/provided/first_frame.png'
    forwarded = []
    for name in Settings.__dataclass_fields__:
        value = getattr(args, 'consistency_' + name)
        if value != '':
            forwarded.extend(['--consistency-' + name.replace('_', '-'), str(value)])
    rows = []
    for model, expected in [('cogvideox1.5-5b-i2v',False), ('hunyuan-video-1.5-i2v',True)]:
        video = source/'data/videos/all_test'/model/'gpt/g2_P19_seed42.mp4'
        output = run/model/'result_g2_P19_seed42.json'
        debug = run/model/'debug'
        code = evaluate('P19', ROOT/'g2/P19', ['--video',str(video),'--image',str(reference),
            '--video_prompt_file',str(source/'g2/P19/prompts/video.txt'),'--output',str(output),
            '--debug-dir',str(debug),'--sample-id','g2_P19_seed42','--model',model,'--seed','42',*forwarded])
        result = json.loads(output.read_text())
        summary = result['verbose']['M1']['_scoring_summary']
        backend_called = (debug/'backend.log').is_file()
        row = {'model':model,'expected_passed':expected,'exit_code':code,'backend_called':backend_called,
               'result':str(output),'summary':summary,'decision':result['verbose']['M1']['_consistency']}
        rows.append(row)
        (run/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        assert code in (0,1), row
        assert summary['consistency_passed'] is expected, row
        assert summary['physics_attempted'] is expected and backend_called is expected, row
        expected_score = 0.15*summary['consistency_score']+0.85*(summary['physics_score'] or 0.0)
        assert math.isclose(summary['score'],expected_score,abs_tol=1e-12), row
    (args.output_root/'latest.json').write_text(json.dumps({'run':str(run),'summary':str(run/'summary.json'),'passed':True},indent=2)+'\n')
    print(json.dumps({'passed':True,'summary':str(run/'summary.json')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

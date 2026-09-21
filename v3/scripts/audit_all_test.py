#!/usr/bin/env python3
"""Audit every current formal result against source inputs, gate and score formula."""
import json
import math
from pathlib import Path
import sys

from all_test_common import ROOT, V3, SOURCE, digest, now, write_json
sys.path.insert(0, str(V3))
from unified_evaluators.consistency import RUBRIC_VERSION


def main():
    manifest = json.loads((ROOT / 'run_manifest.json').read_text())
    problems = []
    checked = 0
    for job in manifest['jobs']:
        prefix = f"{job['model']}/{job['sample_id']}"
        try:
            record = json.loads(Path(job['execution_record']).read_text())
            result = json.loads(Path(job['output']).read_text())
            b = result['verbose']['M1']
            gate, summary, prov = b['_consistency'], b['_scoring_summary'], b['_provenance']
            assert record.get('finished_at_utc'), 'unfinished execution'
            assert record['signature'] == job['signature'], 'changed input signature'
            assert record['result_sha256'] == digest(job['output']), 'result hash mismatch'
            assert set(result) == {'task_id', 'video_path', 'image_path', 'video_prompt', 'model', 'seed', 'metrics', 'verbose'}, 'public schema'
            assert result['task_id'] == job['task'] and result['model'] == job['model'] and result['seed'] == job['seed'], 'sample identity mismatch'
            assert prov['video_sha256'] == job['inputs']['video_sha256'] == digest(job['video']), 'source video hash mismatch'
            assert prov['image_sha256'] == job['inputs']['image_sha256'] == digest(job['image']), 'source image hash mismatch'
            assert gate['rubric_version'] == RUBRIC_VERSION == manifest['rubric_version'], 'mixed consistency rubric'
            assert gate['threshold'] == manifest['threshold'], 'mixed consistency thresholds'
            assert gate['rubric_sha256'] == job['inputs']['rubric_sha256'], 'consistency rubric hash mismatch'
            assert gate['status'] == 'evaluated', 'VLM execution error: ' + gate.get('reason', '')
            assert gate == json.loads((Path(job['debug']) / 'consistency/consistency.json').read_text()), 'gate evidence mismatch'
            assert summary['score_status'] not in {'consistency_error', 'execution_error'}, 'runtime error: ' + str(prov.get('runtime_error'))
            assert gate['passed'] == (gate['score'] >= gate['threshold']), 'gate threshold mismatch'
            passed = gate['passed']
            assert summary['physics_attempted'] == passed, 'gate control flow mismatch'
            expected = .15 * gate['score']
            if passed:
                defined = summary['defined_metrics']
                physics = sum((result['metrics'][k]['metric'] or 0) for k in defined) / len(defined)
                assert math.isclose(physics, summary['physics_score'], abs_tol=1e-12), 'physics aggregation mismatch'
                expected += .85 * physics
                if job['task'] in {'P3', 'P9'}:
                    track = prov['track_cache']
                    assert track['neural_inference_rerun'] is True and track['video_sha256'] == prov['video_sha256'], 'old coordinates reused'
                extraction = prov.get('extraction_run')
                if extraction:
                    assert not extraction.get('sam2_cache_reused') and not extraction.get('cotracker_cache_reused'), 'prior segmentation/tracks reused'
            else:
                assert summary['physics_score'] is None and prov['backend_exit_code'] is None, 'rejected sample ran backend'
                assert not (Path(job['debug']) / 'backend.log').exists(), 'rejected sample has backend execution'
            assert math.isclose(summary['score'], expected, abs_tol=1e-12), 'total score formula mismatch'
            assert math.isclose(record['score'], summary['score'], abs_tol=1e-12), 'execution/result score mismatch'
            for frame in gate['frames']:
                assert Path(frame['path']).is_file(), 'missing VLM input frame'
            checked += 1
        except Exception as exc:
            problems.append({'sample': prefix, 'reason': f'{type(exc).__name__}: {exc}'})
    source_count = len(list((SOURCE / 'data/videos/all_test').glob('*/gpt/*.mp4')))
    audit = {'audited_at_utc': now(), 'checked': checked, 'manifest_jobs': len(manifest['jobs']),
             'source_videos': source_count, 'all_source_videos_included': len(manifest['jobs']) == source_count,
             'passed': not problems and len(manifest['jobs']) == source_count, 'problems': problems}
    write_json(ROOT / 'audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

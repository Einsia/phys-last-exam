#!/usr/bin/env python3
"""Install shared output adapters while retaining every existing measurement backend."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    for metadata in sorted(ROOT.glob('g[1-9]/P*/metadata_v2.json')):
        task = metadata.parent
        entry = task/'evaluator/evaluate.py'
        backup = task/'evaluator/measure_backend.py'
        if not backup.exists():
            shutil.copy2(entry,backup)
        entry.write_text(f'''#!/usr/bin/env python3
"""{task.name}: fresh measurement with the unified G1-G9 V2 output contract."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from unified_evaluators.runtime import main
if __name__ == '__main__':
    raise SystemExit(main('{task.name}', Path(__file__).resolve().parents[1]))
''')
        script = task/'scripts/run_eval.sh'
        script.parent.mkdir(parents=True,exist_ok=True)
        script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
evaluation_root="$(cd -- "$task_dir/../.." && pwd)"
evaluation_python="${{EVALUATOR_PYTHON:-$evaluation_root/.venv/bin/python}}"
model="minimax_h3"
if [[ $# -gt 0 && "$1" != -* ]]; then model="$1"; shift; fi
exec "$evaluation_python" "$evaluation_root/scripts/run_all_eval.py" --groups {task.parent.name} --tasks {task.name} --model "$model" --log-dir "$task_dir/eval_results/run_logs" "$@"
''')
        script.chmod(0o755)
    for group in ROOT.glob('g[1-9]'):
        script = group/'scripts/run_eval.sh'
        script.parent.mkdir(parents=True,exist_ok=True)
        script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
group_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
evaluation_root="$(cd -- "$group_dir/.." && pwd)"
evaluation_python="${{EVALUATOR_PYTHON:-$evaluation_root/.venv/bin/python}}"
model="minimax_h3"
if [[ $# -gt 0 && "$1" != -* ]]; then model="$1"; shift; fi
exec "$evaluation_python" "$evaluation_root/scripts/run_all_eval.py" --groups {group.name} --model "$model" --log-dir "$group_dir/eval_run_logs" "$@"
''')
        script.chmod(0o755)
    # Keep numerical output at full precision and declare the mapping explicitly.
    for schema in ROOT.glob('g[146]/P*/evaluator/utils/physeval/schema.py'):
        text = schema.read_text()
        for before,after in (
            ('round(float(self.value), 6)','float(self.value)'),('round(q, 6)','q'),
            ('round(float(self.score), 6)','float(self.score)'),('round(self.physics_score, 6)','self.physics_score'),
            ('"error_scale_a": self.a,','"error_scale_a": self.a,\n            "ideal": self.ideal,\n            "non_residual": self.non_residual,')):
            if before == '"error_scale_a": self.a,' and '"ideal": self.ideal' in text:
                continue
            text = text.replace(before,after)
        schema.write_text(text)
    print('Installed 40 task entrypoints, 40 task batch scripts and 9 group batch scripts.')


if __name__ == '__main__':
    main()

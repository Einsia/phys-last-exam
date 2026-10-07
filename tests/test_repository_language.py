"""Keep repository text and decoded output strings free of CJK ideographs."""
import ast
import json
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r'[\u2e80-\u2fdf\u3005-\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]')
LOCAL_DIRECTORIES = {'.git', '.venv', '.cache', '__pycache__', '.pytest_cache',
                     'runs', 'videos', 'models', '.models', 'eval_results',
                     'output_videos', 'results', 'work'}


class RepositoryLanguageTests(unittest.TestCase):
    def test_repository_text_and_decoded_strings_use_english(self):
        violations = []
        for directory, folders, files in os.walk(ROOT):
            folders[:] = [name for name in folders if name not in LOCAL_DIRECTORIES]
            for name in files:
                path = Path(directory) / name
                relative = path.relative_to(ROOT).as_posix()
                if HAN.search(relative):
                    violations.append(relative + ': filename')
                try:
                    text = path.read_text(encoding='utf-8-sig')
                except (UnicodeDecodeError, OSError):
                    continue
                if '\0' in text:
                    continue
                if HAN.search(text):
                    violations.append(relative + ': source text')
                if path.suffix == '.py':
                    for node in ast.walk(ast.parse(text)):
                        if isinstance(node, ast.Constant) and isinstance(node.value, str) and HAN.search(node.value):
                            violations.append(f'{relative}:{node.lineno}: decoded string')
                elif path.suffix == '.json':
                    if HAN.search(json.dumps(json.loads(text), ensure_ascii=False)):
                        violations.append(relative + ': decoded JSON')
        self.assertEqual(violations, [], '\n'.join(violations))


if __name__ == '__main__':
    unittest.main()

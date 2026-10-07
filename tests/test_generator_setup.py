"""Check that an interrupted source fetch can be retried cleanly."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('run', scripts / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
spec = importlib.util.spec_from_file_location('ple_generator_setup', scripts / 'prepare_generator.py')
generator = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'run': launcher}):
    spec.loader.exec_module(generator)


class GeneratorSetupTests(unittest.TestCase):
    def test_fetch_failure_leaves_source_absent_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source/Wan2.2'

            def command(arguments, **kwargs):
                if arguments[1] == 'init':
                    Path(arguments[-1]).mkdir()
                if 'fetch' in arguments:
                    raise subprocess.CalledProcessError(128, arguments)

            with patch.object(generator, 'run', side_effect=command):
                with self.assertRaises(subprocess.CalledProcessError):
                    generator.checkout(source, 'https://example.test/repo.git', 'fixture-revision')
            self.assertFalse(source.exists())
            self.assertEqual(list(source.parent.iterdir()), [])

            def success(arguments, **kwargs):
                if arguments[1] == 'init':
                    Path(arguments[-1]).mkdir()

            with patch.object(generator, 'run', side_effect=success):
                generator.checkout(source, 'https://example.test/repo.git', 'fixture-revision')
            self.assertTrue(source.is_dir())


if __name__ == '__main__':
    unittest.main()

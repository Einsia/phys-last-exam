"""Check that explicit configuration errors never fall back to inference defaults."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate


class GenerationConfigTests(unittest.TestCase):
    def test_missing_explicit_config_is_rejected_before_planning(self):
        with tempfile.TemporaryDirectory() as directory, patch('sys.stderr'), \
                patch.object(generate, 'plan_jobs') as plan:
            with self.assertRaises(SystemExit) as raised:
                generate.main(['--models', 'cogvideox1.5-5b-i2v', '--config',
                               str(Path(directory) / 'missing.json'), '--dry-run'])
            self.assertEqual(raised.exception.code, 2)
            plan.assert_not_called()


if __name__ == '__main__':
    unittest.main()

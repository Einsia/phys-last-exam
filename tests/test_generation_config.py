"""Regression checks for GPU selection in reusable local generator profiles."""
import tempfile
import unittest

from generation.backends import REGISTRY
from generation.config import default_config, resolve_profile


class DeviceOverrideTests(unittest.TestCase):
    def test_override_selects_requested_gpu_without_mutating_saved_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            name = 'cogvideox1.5-5b-i2v'
            saved = default_config(directory)['models'][name]
            profile = resolve_profile(name, saved, directory, devices='3')
            runner = REGISTRY[name](**profile['options'])
            self.assertEqual(runner._env()['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(saved['options']['devices'], '0')

    def test_lingbot_gpu_override_matches_distributed_launch_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            name = 'lingbot-video-moe-30b-a3b'
            saved = default_config(directory)['models'][name]
            for devices, count in [('3', 1), ('1,2', 2)]:
                with self.subTest(devices=devices):
                    profile = resolve_profile(name, saved, directory, devices=devices)
                    options = profile['options']
                    self.assertEqual(options['nproc_per_node'], count)
                    self.assertEqual(options['context_parallel_degree'], count)
                    for flag in ('distributed', 'enable_fsdp_inference', 'enable_vlm_fsdp_inference'):
                        self.assertEqual(options[flag], count > 1)
                    self.assertEqual(REGISTRY[name](**options)._env()['CUDA_VISIBLE_DEVICES'], devices)


if __name__ == '__main__':
    unittest.main()

"""Regression checks for portable generation manifests (no model inference)."""
import base64
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import evaluate as batch
from generate import update_manifest


class GenerationManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / 'task'
        self.task.mkdir()
        image = self.task / 'first_frame.png'
        image.write_bytes(base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/aUAAAAASUVORK5CYII='))
        prompt = self.task / 'prompt.txt'
        prompt.write_text('A controlled scene.\n', encoding='utf-8')
        template = self.task / 'first_frame_annotations.json'
        template.write_text('{"kind": "template"}\n', encoding='utf-8')
        self.output = self.root / 'videos'
        self.video = self.output / 'fixture-cpu' / 'g8_P33_seed42.mp4'
        self.video.parent.mkdir(parents=True)
        # Inventory checks hash the file; decoding/inference is outside these tests.
        self.video.write_bytes(b'unchanged video fixture')
        self.job = dict(model='fixture-cpu', task='P33', seed=42,
                        sample_id='g8_P33_seed42', video=str(self.video),
                        image=str(image), prompt_file=str(prompt),
                        annotation_template=str(template),
                        input_hashes=dict(image=batch.sha(image), prompt=batch.sha(prompt),
                                          annotation=batch.sha(template)))
        self.overrides = self.root / 'external-annotations'
        self.annotation = self.overrides / 'fixture-cpu' / 'P33' / 'g8_P33_seed42.json'
        self.annotation.parent.mkdir(parents=True)
        self.annotation.write_text('{"kind": "reviewed override", "revision": 1}\n', encoding='utf-8')

    def manifest(self):
        update_manifest(self.output, self.root / 'work', [self.job], self.overrides)
        return batch.read(self.output / 'manifest.json')

    def test_custom_annotation_survives_moving_only_videos(self):
        original = self.annotation.read_bytes()
        rows = self.manifest()
        snapshot = (self.output / rows[0]['annotation']).resolve()
        self.assertTrue(snapshot.is_relative_to(self.output.resolve()))
        self.assertEqual(snapshot.read_bytes(), original)
        portable = self.root / 'another-machine' / 'videos'
        shutil.copytree(self.output, portable)
        self.annotation.unlink()
        jobs, _ = batch.prepare_jobs(rows, portable, {'P33': {'path': str(self.task)}})
        self.assertEqual(jobs[0]['input_errors'], [])
        self.assertEqual(Path(jobs[0]['annotation']).read_bytes(), original)

    def test_changed_override_creates_new_snapshot_without_changing_video(self):
        first = self.manifest()[0]
        original_snapshot = (self.output / first['annotation']).read_bytes()
        original_video_hash = batch.sha(self.video)
        self.annotation.write_text('{"kind": "reviewed override", "revision": 2}\n', encoding='utf-8')
        second = self.manifest()[0]
        self.assertNotEqual(first['annotation'], second['annotation'])
        self.assertEqual((self.output / first['annotation']).read_bytes(), original_snapshot)
        self.assertEqual((self.output / second['annotation']).read_bytes(), self.annotation.read_bytes())
        self.assertEqual(batch.sha(self.video), original_video_hash)

    def test_corrupt_existing_snapshot_is_rejected(self):
        row = self.manifest()[0]
        (self.output / row['annotation']).write_text('corrupt', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'annotation snapshot is corrupt'):
            self.manifest()


if __name__ == '__main__':
    unittest.main()

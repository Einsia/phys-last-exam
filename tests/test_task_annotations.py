"""Reusable task templates must bind to real frames, without weakening overrides."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'medium/P37/evaluator/_shared'))
import evaluate
from refined_evaluators.annotations import TEMPLATE_TYPE, scale_annotation
from refined_evaluators.common import ExtractionError
from refined_evaluators.media import fingerprint, frame_fingerprint
from refined_evaluators.runtime import parser
from refined_evaluators.vision import load_annotation

TASKS = ('P37', 'P38', 'P39', 'P41', 'P43', 'P49')


def write_clip(path, first, later_offset=0):
    with av.open(str(path), 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height = first.shape[1::-1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for index in range(3):
            data = first if index == 0 else np.clip(first.astype(float) + later_offset, 0, 255).astype('uint8')
            frame = av.VideoFrame.from_ndarray(data, format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format='bgr24') for frame in container.decode(video=0)]


class TaskAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.tasks = evaluate.discover_tasks()

    def inputs(self, task='P37', size=(832, 480), suffix='default', later_offset=0):
        folder = Path(self.tasks[task]['path'])
        video = self.directory / (suffix + '.mp4')
        first = cv2.resize(cv2.imread(str(folder / 'first_frame.png')), size)
        frames = write_clip(video, first, later_offset)
        args = SimpleNamespace(task_id=task, image_path=str(folder / 'first_frame.png'),
                               video_path=str(video), annotation=str(folder / 'first_frame_annotations.json'))
        debug = self.directory / (suffix + '-debug')
        debug.mkdir()
        return args, frames, debug

    def test_all_six_templates_are_bound_to_packaged_images(self):
        found = []
        for task, info in self.tasks.items():
            path = Path(info['path']) / 'first_frame_annotations.json'
            if not path.exists():
                continue
            found.append(task)
            template = evaluate.read(path)
            self.assertEqual(template['annotation_type'], TEMPLATE_TYPE)
            self.assertNotIn('source_video_sha256', template)
            self.assertEqual(template['source_image_sha256'], fingerprint(path.with_name('first_frame.png')))
            self.assertTrue(template['template_review']['accepted'])
        self.assertEqual(set(found), set(TASKS))

    def test_all_tasks_reuse_templates_on_two_encoded_video_resolutions(self):
        for task in TASKS:
            for size in ((832, 480), (1344, 768)):
                with self.subTest(task=task, size=size):
                    args, frames, debug = self.inputs(task, size, task + '-' + str(size[0]))
                    annotation = load_annotation(args, frames, debug)
                    self.assertEqual(annotation['size_wh'], list(size))
                    self.assertEqual(annotation['source_video_sha256'], fingerprint(args.video_path))
                    self.assertEqual(annotation['frame0_sha256'], frame_fingerprint(frames[0]))
                    self.assertTrue(annotation['image_correspondence']['accepted'])
                    self.assertTrue((debug / 'initial_reference.json').is_file())
                    self.assertTrue((debug / 'first_frame_identity.png').is_file())

    def test_one_task_template_binds_to_two_different_videos(self):
        bound = []
        for offset in (0, 20):
            args, frames, debug = self.inputs(suffix=str(offset), later_offset=offset)
            bound.append(load_annotation(args, frames, debug))
        self.assertEqual(bound[0]['source_image_sha256'], bound[1]['source_image_sha256'])
        self.assertNotEqual(bound[0]['source_video_sha256'], bound[1]['source_video_sha256'])
        self.assertEqual(bound[0]['template_binding'], bound[1]['template_binding'])

    def test_geometry_uses_both_coordinate_scales_without_changing_directions(self):
        folder = Path(self.tasks['P49']['path'])
        scaled = scale_annotation(evaluate.read(folder / 'first_frame_annotations.json'), [832, 480])
        self.assertEqual(scaled['geometry']['bottom_y'], 440)
        self.assertEqual(scaled['geometry']['tank_walls_x'], [325, 506])
        self.assertEqual(scaled['geometry']['down_axis'], [0, 1])
        self.assertEqual(scaled['objects'][0]['positive'], [[459, 106]])
        folder = Path(self.tasks['P39']['path'])
        scaled = scale_annotation(evaluate.read(folder / 'first_frame_annotations.json'), [832, 480])
        self.assertEqual(scaled['geometry']['entry_plane'], [[301, 165], [301, 254]])
        self.assertEqual(scaled['geometry']['lamp_emission_box'], [698, 238, 722, 269])

    def test_changed_first_frame_fails_with_alignment_evidence(self):
        args, frames, debug = self.inputs()
        # Keep the exact task input and video binding, but remove the objects
        # from the decoded frame supplied to initialization.
        frames[0][:] = 0
        with self.assertRaisesRegex(ExtractionError, 'does not match the generated first frame'):
            load_annotation(args, frames, debug)
        evidence = evaluate.read(debug / 'annotation_alignment.json')
        self.assertFalse(evidence['image_correspondence']['accepted'])
        self.assertFalse((debug / 'initial_reference.json').exists())

    def test_different_input_image_or_task_cannot_reuse_template(self):
        args, frames, debug = self.inputs()
        other = Path(self.tasks['P39']['path']) / 'first_frame.png'
        args.image_path = str(other)
        with self.assertRaisesRegex(ExtractionError, 'image hash mismatch'):
            load_annotation(args, frames, debug)
        args.task_id = 'P39'
        with self.assertRaisesRegex(ExtractionError, 'task ID'):
            load_annotation(args, frames, debug)

    def test_custom_video_annotation_retains_video_hash_guard(self):
        args, frames, debug = self.inputs(size=(1344, 768))
        custom = evaluate.read(args.annotation)
        custom.update(annotation_type='agent_visual_first_frame_review',
                      coordinate_frame='decoded_video_frame_0', source_video_sha256='different-video')
        path = self.directory / 'custom.json'
        path.write_text(json.dumps(custom))
        args.annotation = str(path)
        with self.assertRaisesRegex(ExtractionError, 'video hash mismatch'):
            load_annotation(args, frames, debug)
        custom['source_video_sha256'] = fingerprint(args.video_path)
        path.write_text(json.dumps(custom))
        self.assertTrue(load_annotation(args, frames, debug)['image_correspondence']['accepted'])

    def test_batch_uses_task_defaults_and_explicit_overrides(self):
        video = self.directory / 'video.mp4'
        video.write_bytes(b'inventory fixture')
        row = dict(model='fixture', task='P37', sample_id='g8_P37_seed42', video=str(video))
        jobs, _ = evaluate.prepare_jobs([row], self.directory, self.tasks)
        self.assertEqual(Path(jobs[0]['annotation']), Path(self.tasks['P37']['path']) / 'first_frame_annotations.json')
        self.assertFalse(jobs[0]['input_errors'])
        self.assertIn('annotation', jobs[0]['files'])
        overrides = self.directory / 'annotations'
        override = overrides / 'fixture/P37/g8_P37_seed42.json'
        override.parent.mkdir(parents=True)
        override.write_text('{}')
        row['annotation'] = jobs[0]['annotation']  # A generated manifest includes the template.
        jobs, _ = evaluate.prepare_jobs([row], self.directory, self.tasks, annotation_root=overrides)
        self.assertEqual(jobs[0]['annotation'], str(override))

    def test_standalone_parser_defaults_to_its_bundled_template(self):
        args = parser('P37').parse_args(['--video_path', 'new.mp4', '--output', 'result.json'])
        self.assertEqual(Path(args.annotation), ROOT / 'medium/P37/first_frame_annotations.json')
        self.assertEqual(args.task_id, 'P37')
        custom = parser('P37').parse_args(['--video_path', 'new.mp4', '--output', 'result.json', '--annotation', 'custom.json'])
        self.assertEqual(custom.annotation, 'custom.json')

    def test_standalone_packages_share_the_complete_loader(self):
        for name in ('annotations.py', 'vision.py', 'runtime.py'):
            files = list(ROOT.glob('*/P*/evaluator/_shared/refined_evaluators/' + name))
            self.assertEqual(len(files), 8)
            self.assertEqual(len({file.read_bytes() for file in files}), 1)

    def test_unknown_geometry_is_not_silently_left_unscaled(self):
        template = evaluate.read(ROOT / 'medium/P37/first_frame_annotations.json')
        template['geometry']['unrecognized_geometry'] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ExtractionError, 'Unknown template geometry field'):
            scale_annotation(template, [832, 480])


if __name__ == '__main__':
    unittest.main()

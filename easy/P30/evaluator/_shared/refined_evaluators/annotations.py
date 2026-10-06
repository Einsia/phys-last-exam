"""Reuse reviewed task-image annotations after checking the actual video frame."""
from copy import deepcopy

import cv2
import numpy as np

from .common import ExtractionError, write_json
from .media import fingerprint, frame_fingerprint

TEMPLATE_TYPE = 'task_first_frame_template'
MAX_OBJECT_DIFFERENCE = 15.0  # Same threshold as the existing image/video check.


def scale_annotation(annotation, size):
    """Map the declared annotation canvas through a full-image resize."""
    result = deepcopy(annotation)
    original = np.asarray(annotation['size_wh'], dtype=float)
    target = np.asarray(size, dtype=float)
    if original.shape != (2,) or not np.isfinite(original).all() or np.any(original <= 0):
        raise ExtractionError('Invalid annotation canvas size')
    scale = target / original

    def points(value):
        data = np.asarray(value, dtype=float)
        if data.size == 0:
            return []
        if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
            raise ExtractionError('Invalid annotation points')
        return np.rint(data * scale).astype(int).tolist()

    def box(value):
        data = np.asarray(value, dtype=float)
        if data.shape != (4,) or not np.isfinite(data).all():
            raise ExtractionError('Invalid annotation box')
        return np.rint(data * np.tile(scale, 2)).astype(int).tolist()

    for obj in result['objects']:
        obj['box'] = box(obj['box'])
        obj['positive'] = points(obj['positive'])
        if 'negative' in obj:
            obj['negative'] = points(obj['negative'])
    point_keys = {'pivot_points', 'entry_plane', 'exit_plane', 'outlet_points', 'ground_points'}
    box_keys = {'gap_evidence_box', 'lamp_emission_box', 'lamp_background_box'}
    box_list_keys = {'fixed_reference_boxes', 'stream_boxes'}
    vector_keys = {'up_axis', 'down_axis', 'pass_direction'}
    for key, value in result['geometry'].items():
        if key in point_keys:
            result['geometry'][key] = points(value)
        elif key in box_keys:
            result['geometry'][key] = box(value)
        elif key in box_list_keys:
            result['geometry'][key] = [box(item) for item in value]
        elif key in {'bottom_y', 'liquid_surface_y'}:
            result['geometry'][key] = int(round(value * scale[1]))
        elif key == 'tank_walls_x':
            result['geometry'][key] = [int(round(x * scale[0])) for x in value]
        elif key in vector_keys:
            vector = np.asarray(value, dtype=float) * scale
            length = np.linalg.norm(vector)
            if vector.shape != (2,) or not np.isfinite(length) or length == 0:
                raise ExtractionError('Invalid annotation direction')
            result['geometry'][key] = (vector / length).tolist()
        elif key not in {'axisymmetric', 'release_reference'}:
            raise ExtractionError(f'Unknown template geometry field: {key}')
    result['size_wh'] = list(size)
    return result


def validate_objects(annotation, size):
    width, height = size
    names = set()
    if not annotation['objects']:
        raise ExtractionError('Task annotation has no objects')
    for obj in annotation['objects']:
        if obj['name'] in names or not obj.get('identity_evidence'):
            raise ExtractionError('Task annotation lacks distinct reviewed object identities')
        names.add(obj['name'])
        x1, y1, x2, y2 = obj['box']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ExtractionError('Scaled annotation object box lies outside the video')
        if not obj.get('positive'):
            raise ExtractionError('Task annotation has no material point')
        if any(not (0 <= x < width and 0 <= y < height)
               for x, y in obj['positive'] + obj.get('negative', [])):
            raise ExtractionError('Scaled annotation point lies outside the video')


def bind_task_template(template, args, first_frame, directory, template_path):
    """Bind a reusable image template to this video's measured frame and hashes."""
    if template.get('template_version') != 1 or template.get('coordinate_frame') != 'resized_input_image':
        raise ExtractionError('Unsupported task annotation template format')
    if template.get('task_id') != getattr(args, 'task_id', template.get('task_id')):
        raise ExtractionError('Annotation task ID does not match the evaluator')
    review = template.get('template_review', {})
    if review.get('accepted') is not True or not review.get('evidence'):
        raise ExtractionError('Task annotation has no accepted input-image review')
    if not args.image_path or template['source_image_sha256'] != fingerprint(args.image_path):
        raise ExtractionError('Task annotation image hash mismatch; supply --annotation for a different input image')
    still = cv2.imread(str(args.image_path))
    if still is None or list(still.shape[1::-1]) != template.get('source_size_wh'):
        raise ExtractionError('Task annotation source image is unreadable or has different dimensions')
    size = list(first_frame.shape[1::-1])
    annotation = scale_annotation(template, size)
    validate_objects(annotation, size)
    reference = cv2.GaussianBlur(cv2.resize(still, tuple(size)), (5, 5), 0).astype(np.float32)
    actual = cv2.GaussianBlur(first_frame, (5, 5), 0).astype(np.float32)
    errors = []
    for obj in annotation['objects']:
        x1, y1, x2, y2 = obj['box']
        errors.append(float(np.mean(np.abs(reference[y1:y2, x1:x2] - actual[y1:y2, x1:x2]))))
    check = dict(accepted=max(errors) < MAX_OBJECT_DIFFERENCE,
                 method='task_image_hash_and_resized_object_crop_difference',
                 mean_absolute_difference=float(np.mean(np.abs(reference - actual))),
                 object_crop_errors=errors, object_names=[o['name'] for o in annotation['objects']],
                 object_difference_threshold=MAX_OBJECT_DIFFERENCE)
    annotation.update(coordinate_frame='decoded_video_frame_0',
                      source_video_sha256=fingerprint(args.video_path),
                      frame0_sha256=frame_fingerprint(first_frame), image_correspondence=check,
                      template_binding=dict(template_sha256=fingerprint(template_path),
                                            source_canvas_wh=template['size_wh'], target_size_wh=size,
                                            transform='full_image_resize',
                                            scale_xy=[size[i] / template['size_wh'][i] for i in range(2)]))
    write_json(directory / 'annotation_alignment.json', annotation)
    if not check['accepted']:
        raise ExtractionError('Task first-frame annotation does not match the generated first frame; '
                              'see annotation_alignment.json. Changed layouts need a reviewed --annotation override.')
    return annotation

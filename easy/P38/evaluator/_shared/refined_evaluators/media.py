"""Timestamp-preserving decoding and intermediate video export."""
from fractions import Fraction
from pathlib import Path
import hashlib
import numpy as np
import av
import cv2
from .common import ExtractionError, EnvironmentError, write_json


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frame_fingerprint(frame):
    """Bind initialization to the actual decoded BGR frame, including its shape."""
    value = np.ascontiguousarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Expected a decoded uint8 BGR image')
    h = hashlib.sha256()
    h.update(str(value.shape).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def decode(path):
    frames, times, pts = [], [], []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ExtractionError('Missing video PTS/time base')
                t = float(frame.pts * frame.time_base)
                if times and t <= times[-1]:
                    raise ExtractionError('Non-increasing video presentation timestamps')
                frames.append(frame.to_ndarray(format='bgr24'))
                times.append(t)
                pts.append({'pts': frame.pts, 'time_base': str(frame.time_base), 'time_sec': t})
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f'Cannot decode video: {e}') from e
    if len(frames) < 3 or any(f.shape != frames[0].shape for f in frames):
        raise ExtractionError('Insufficient frames or changing video dimensions')
    return frames, np.array(times), pts


class VideoWriter:
    """H.264 for viewing; every output frame carries its source presentation time."""
    def __init__(self, path, width, height, times):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.origin = float(times[0])
        rate = Fraction(float(1 / np.median(np.diff(times)))).limit_denominator(100000)
        try:
            self.container = av.open(str(path), 'w')
            self.stream = self.container.add_stream('libx264', rate=rate)
            self.stream.width = width + width % 2
            self.stream.height = height + height % 2
            self.stream.pix_fmt = 'yuv420p'
            self.stream.time_base = Fraction(1, 1000000)
            self.stream.codec_context.time_base = self.stream.time_base
            self.stream.options = {'crf': '16', 'preset': 'fast', 'bf': '0'}
        except Exception as e:
            if hasattr(self, 'container'):
                self.container.close()
            raise EnvironmentError(f'H.264 encoder unavailable: {e}') from e

    def write(self, image, time_sec):
        h, w = image.shape[:2]
        image = cv2.copyMakeBorder(image, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts = round((float(time_sec) - self.origin) * 1000000)
        frame.time_base = Fraction(1, 1000000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def save_segments(frames, times, masks, boxes, directory):
    """Export before measurement, so even a failed extraction has visual evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    crops, writers, artifacts = [], [], {}
    try:
        for j, name in enumerate(('left', 'right')):
            x1, y1, x2, y2 = boxes[j]
            pad = 0.15 * max(x2 - x1, y2 - y1)
            crop = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                    min(w, int(x2 + pad)), min(h, int(y2 + pad)))
            crops.append(crop)
            cw, ch = crop[2] - crop[0], crop[3] - crop[1]
            for suffix in ('compass', 'mask'):
                path = directory / f'{name}_{suffix}.mp4'
                writers.append(VideoWriter(path, cw, ch, times))
                artifacts[f'{name}_{suffix}'] = str(path)
        overlay = VideoWriter(directory / 'segmentation_overlay.mp4', w, h, times)
        writers.append(overlay)
        artifacts['segmentation_overlay'] = str(overlay.path)
        for i, (frame, t) in enumerate(zip(frames, times)):
            canvas = frame.copy()
            for j, color in enumerate(((50, 220, 50), (255, 160, 20))):
                mask = masks[i, j]
                x1, y1, x2, y2 = crops[j]
                cutout = np.where(mask[..., None], frame, 0).astype(np.uint8)
                binary = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
                writers[2*j].write(cutout[y1:y2, x1:x2], t)
                writers[2*j+1].write(binary[y1:y2, x1:x2], t)
                canvas[mask] = (0.72 * canvas[mask] + 0.28 * np.array(color)).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2)
                cv2.putText(canvas, ('left', 'right')[j], (x1, max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
            cv2.putText(canvas, f't={t:.3f}s', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            overlay.write(canvas, t)
    finally:
        for writer in writers:
            writer.close()
    write_json(directory / 'crop_coordinates.json', {'crops_xyxy': crops, 'source_size_wh': [w, h],
               'note': 'MP4 masks are viewing copies. masks.npz contains exact binary masks.'})
    return artifacts

"""Validate video output and prepare the fixed P3/P9 measurement canvas."""
from fractions import Fraction
from pathlib import Path
import shutil


def inspect_video(path):
    import av
    with av.open(str(path)) as container:
        if 'mp4' not in container.format.name.split(','):
            raise ValueError('Generated file must use an MP4 container, not just an .mp4 filename')
        if not container.streams.video:
            raise ValueError('Generated file has no video stream')
        stream = container.streams.video[0]
        count, previous, first = 0, None, None
        size = None
        for frame in container.decode(stream):
            current_size = (frame.width, frame.height)
            if size is not None and size != current_size:
                raise ValueError('Video resolution changes between frames')
            size = current_size
            if frame.time is None or (previous is not None and frame.time <= previous):
                raise ValueError('Video timestamps are missing or non-monotonic')
            first = frame.time if first is None else first
            previous = frame.time
            count += 1
        if count < 2 or not stream.average_rate or stream.average_rate <= 0:
            raise ValueError('Generated video must contain at least two timed frames')
        return dict(width=size[0], height=size[1], frames=count, fps=float(stream.average_rate),
                    rate=str(stream.average_rate), first_time=first, last_time=previous)


def prepare_video(raw, output, task):
    import av
    raw, output = Path(raw), Path(output)
    info = inspect_video(raw)
    resize = task in {'P3', 'P9'} and (info['width'], info['height']) != (1344, 768)
    if resize:
        with av.open(str(raw)) as source, av.open(str(output), 'w', format='mp4') as destination:
            stream = destination.add_stream('libx264', rate=Fraction(info['rate']))
            stream.width, stream.height, stream.pix_fmt = 1344, 768, 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast'}
            for frame in source.decode(video=0):
                resized = frame.reformat(width=1344, height=768, format='yuv420p')
                resized.pts, resized.time_base = frame.pts, frame.time_base
                for packet in stream.encode(resized):
                    destination.mux(packet)
            for packet in stream.encode():
                destination.mux(packet)
    else:
        shutil.copyfile(raw, output)
    exported = inspect_video(output)
    if exported['frames'] != info['frames'] or abs((exported['last_time'] - exported['first_time']) -
                                                (info['last_time'] - info['first_time'])) > 0.002:
        raise ValueError('Export changed the frame count or video timing')
    return dict(raw=info, exported=exported, transform='resize_xy' if resize else 'none',
                scale_xy=[exported['width'] / info['width'], exported['height'] / info['height']],
                raw_video=str(raw), audio='retained in raw video; omitted from resized evaluation copy' if resize else 'unchanged')

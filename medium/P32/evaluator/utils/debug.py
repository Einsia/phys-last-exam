"""Persist per-frame measurement data and visual diagnostics, including failures."""
from pathlib import Path
import csv
import os
import numpy as np
import cv2
from .common import clean_json, write_json
from .media import VideoWriter


def save_measurements(frames, times, rows, calibrations, verbose, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'tracks.json', rows)
    write_json(directory/'initial_reference.json',verbose.get('initial_reference'))
    write_json(directory/'calibration.json', {'method': 'circular dial conic + independently detected physical pivot',
                                            'left': calibrations[0], 'right': calibrations[1]})
    fields = ['frame_index', 'time_sec']
    for name in ('left', 'right'):
        fields.extend(f'{name}_{key}' for key in ('valid','angle_deg','raw_angle_deg','unwrapped_angle_deg',
                                                 'pivot_x','pivot_y','tip_x','tip_y','reason'))
    with (directory/'angles.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in ('frame_index','time_sec')}
            for name in ('left','right'):
                obs = row[name]
                for key in ('valid','angle_deg','raw_angle_deg','unwrapped_angle_deg','reason'):
                    record[f'{name}_{key}'] = obs.get(key)
                for key in ('pivot','tip'):
                    coords = obs.get(key)
                    for axis, index in (('x',0),('y',1)):
                        record[f'{name}_{key}_{axis}'] = float(coords[index]) if coords is not None else None
            writer.writerow(record)
    height, width = frames[0].shape[:2]
    writer = VideoWriter(directory/'needle_tracking.mp4', width, height, times)
    try:
        for frame, row in zip(frames, rows):
            canvas = frame.copy()
            for j, name in enumerate(('left','right')):
                obs = row[name]
                color = ((40,220,40),(255,170,40))[j]
                if calibrations[j] is not None:
                    cv2.ellipse(canvas,calibrations[j]['ellipse'],color,1)
                text = f'{name}: INVALID'
                if obs['valid']:
                    pivot, tip = tuple(np.rint(obs['pivot']).astype(int)), tuple(np.rint(obs['tip']).astype(int))
                    cv2.circle(canvas, pivot, 4, (0,255,255), -1)
                    cv2.arrowedLine(canvas, pivot, tip, color, 2, tipLength=.15)
                    cv2.circle(canvas, tip, 4, (0,0,255), 1)
                    text = f"{name}: {obs['angle_deg']:.2f} deg"
                cv2.putText(canvas, text, (20,65+30*j), cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            cv2.putText(canvas, f"t={row['time_sec']:.3f}s", (20,30), cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            writer.write(canvas,row['time_sec'])
            if row['frame_index'] == 0:
                cv2.imwrite(str(directory/'tracking_first_frame.png'),canvas)
    finally:
        writer.close()
    # Keep font cache inside the output directory rather than changing the user's environment.
    os.environ.setdefault('MPLCONFIGDIR', str(directory.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True)
    for name,color in (('left','tab:green'),('right','tab:blue')):
        measured = [r[name].get('angle_deg') for r in rows]
        unfolded = [r[name].get('unwrapped_angle_deg') for r in rows]
        axes[0].plot(times, measured, '.-', color=color, label=name)
        axes[1].plot(times, unfolded, '.-', color=color, label=name)
    for phase,window in verbose.get('windows',{}).items():
        for ax in axes:
            ax.axvspan(window['start_time_sec'],window['end_time_sec'],color='grey',alpha=.15)
            ax.axvline(window['start_time_sec'],color='grey',linestyle=':',linewidth=.7)
        axes[0].text(window['start_time_sec'],1.02,f"{phase}: {'stable' if window['stable'] else 'unstable'}",
                     transform=axes[0].get_xaxis_transform(),fontsize=9)
    initial = verbose.get('initial_reference')
    if initial:
        for name,color in (('left','tab:green'),('right','tab:blue')):
            for ax in axes:
                ax.scatter([initial['time_sec']],[initial['needles'][name]['angle_deg']],
                           marker='x',s=65,color=color,zorder=5,label=f'{name} initial')
        axes[0].text(initial['time_sec'],1.02,'initial reference (no stillness required)',
                     transform=axes[0].get_xaxis_transform(),fontsize=9)
    axes[0].set_ylabel('Signed angle (deg)')
    axes[1].set_ylabel('Continuous angle (deg)')
    axes[1].set_xlabel('Source presentation time (s)')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('P32 red-pole tracking; CCW positive')
    fig.tight_layout()
    fig.savefig(directory/'angles.png',dpi=160)
    plt.close(fig)
    return {name: str(directory/file) for name,file in (
        ('needle_tracking','needle_tracking.mp4'),('tracks','tracks.json'),('angles_csv','angles.csv'),
        ('angle_plot','angles.png'),('calibration','calibration.json'),('tracking_first_frame','tracking_first_frame.png'),
        ('initial_reference','initial_reference.json'))}

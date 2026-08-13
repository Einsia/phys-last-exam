import sys; sys.path.insert(0, "/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/physical-bench")
import numpy as np
from physbench.render_synth import build
from physbench.tracking import CLASSIC_BACKENDS, run_tracks, estimate_camera_shift
from physbench.kinematics import compensate, extract_flight, _moving_run

for mode, blur in [("gt",1), ("gt",2), ("time_warp",1)]:
    frames, seed, traj = build(mode, theta_deg=45.0, n_flight=60, blur_substeps=blur)
    t = run_tracks(frames, seed, backends=CLASSIC_BACKENDS).tracks["color"]
    shift = estimate_camera_shift(frames, t, seed)
    xy = compensate(t.xy, shift)
    idx = np.flatnonzero(np.isfinite(xy[:,0])); tf = idx.astype(float)
    x, y = xy[idx,0], xy[idx,1]
    vx, vy = np.gradient(x, tf), np.gradient(y, tf)
    sp = np.hypot(vx, vy); ka = int(np.argmin(y))
    peak = float(np.nanmax(sp)); thr = max(0.18*peak, 0.015*seed.radius)
    k0, k1 = _moving_run(sp, ka, thr)
    f, r = extract_flight(xy, seed.radius)
    print(f"== {mode} blur={blur}  truth launch={traj.launch} land={traj.land}")
    print(f"   peak_speed={peak:.2f} thr={thr:.2f} k_apex={ka} (frame {idx[ka]}) k0={k0}(f{idx[k0]}) k1={k1}(f{idx[k1]})")
    print(f"   drift_max={np.max(np.linalg.norm(shift,axis=1)):.3f}px  speed[k0-2:k0+3]={np.round(sp[max(0,k0-2):k0+3],2)}")
    print(f"   speed[k1-2:k1+3]={np.round(sp[max(0,k1-2):k1+3],2)}")
    if f: print(f"   t_launch={f.t_launch:.2f} t_apex={f.t_apex:.2f} t_land={f.t_land:.2f} y_ref={f.y_ref:.2f} y_apex={f.y_apex:.2f} extrap={f.land_extrapolated}")
    else: print(f"   FAIL {r}")
    print(f"   y[k1-3:k1+4]={np.round(y[max(0,k1-3):k1+4],1)}")

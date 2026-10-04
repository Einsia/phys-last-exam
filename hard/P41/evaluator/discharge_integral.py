"""Estimate a free discharge exponent from observed volume and head.

The optional initial waiting period is identified from volume observations before
any exponent is fitted.  Neither interval selection nor numerical search uses
water/sand target exponents.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar


def _observed_discharge_start(volume, t, window, snr):
    """Trim only an initially unresolved waiting period before a sustained drop.

    A decreasing first observation window is kept. Otherwise the initial median
    and MAD define an observation-noise threshold; a sustained volume decrease
    starts the measured interval at the preceding observed reference frame.
    This rule retains an observed boundary reference and never
    searches candidate windows using an exponent or goodness of fit.
    """
    initial = np.flatnonzero(t <= t[0] + window + 1e-9)
    info = {'method': 'initial_volume_plateau_then_sustained_observed_drop',
            'window_sec': float(window), 'snr': float(snr), 'trimmed_frames': 0}
    if len(initial) < 3:
        info['reason'] = 'too_few_initial_observations_to_trim'
        return 0, info
    x = t[initial] - np.mean(t[initial])
    y = volume[initial]
    denominator = float(x @ x)
    slope = float(x @ (y - y.mean()) / denominator)
    residual = y - y.mean() - slope * x
    se = float(np.sqrt((residual @ residual) / max(1, len(x) - 2) / denominator))
    info.update(initial_volume_slope=slope, initial_slope_se=se)
    if -slope > snr * se:
        info['reason'] = 'discharge_already_observed_in_initial_window'
        return 0, info
    baseline = float(np.median(y))
    floor = np.finfo(float).eps * max(1., float(np.max(np.abs(volume)))) * 100
    noise = max(float(1.4826 * np.median(np.abs(y - baseline))), floor)
    info.update(initial_volume_median=baseline, initial_volume_noise_mad=noise,
                required_volume_drop=float(snr * noise))
    for i in range(int(initial[-1]), len(t)):
        if t[-1] - t[i] < window - 1e-9:
            break
        ids = (t >= t[i]) & (t <= t[i] + window + 1e-9)
        if np.mean(volume[ids] < baseline - snr * noise) >= .8:
            crossing = np.flatnonzero(ids & (volume < baseline - snr * noise))[0]
            start = max(0, int(crossing) - 1)
            info.update(trimmed_frames=start, threshold_crossing_sec=float(t[i]),
                        selected_start_sec=float(t[start]),
                        reason='sustained_volume_drop_after_unresolved_initial_window')
            return start, info
    info['reason'] = 'onset_unresolved_keep_all_observations'
    return 0, info


def fit_integrated_discharge(h, volume, t, cfg):
    h, v, t = [np.asarray(x, float) for x in (h, volume, t)]
    good = np.isfinite(h) & np.isfinite(v) & np.isfinite(t) & (h > 0) & (v >= 0)
    ids = np.flatnonzero(good)
    result = {
        'selection_rule': 'Finite observed positive head and volume; optionally remove an initial unresolved waiting period using only sustained observed volume change; fit integrated Q=k*h**beta with a free exponent',
        'independent_fit_frames': [],
        'frame_selection_reasons': ['observed' if ok else ('nonfinite_observation' if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) else 'nonpositive_head' if a <= 0 else 'negative_volume') for a, b, c, ok in zip(h, v, t, good)],
        'fit_window_sec': None, 'fit': None, 'failures': [], 'failure_codes': [],
        'warnings': [],
    }

    def fail(code, message):
        result['failure_codes'].append(code)
        result['failures'].append(message)
        return result

    if len(ids) < 3:
        return fail('insufficient_surfaces', 'too few observed surfaces')
    lo, hi = int(ids[0]), int(ids[-1])
    if not good[lo:hi + 1].all():
        return fail('unbridged_observations', 'unbridged missing volume/head observations')
    if np.any(np.diff(t[lo:hi + 1]) <= 0):
        return fail('nonincreasing_timestamps', 'observed timestamps are not strictly increasing')
    initial_volume = v[lo:hi + 1]
    epsilon = np.finfo(float).eps * max(1., float(np.max(np.abs(initial_volume))))
    if np.ptp(initial_volume) <= epsilon:
        return fail('no_volume_change', 'no observed volume change')
    if np.ptp(np.log(h[lo:hi + 1])) <= 1e-8:
        return fail('constant_head', 'constant observed head: exponent is unidentifiable')
    window = float(cfg['derivative_window_sec'])
    if not np.isfinite(window) or window <= 0:
        return fail('invalid_window', 'observation window must be positive and finite')
    # A positive-discharge law cannot explain a sustained net increase. Do not
    # relabel this as an exponent-search failure or silently reverse its sign.
    begin = t[lo:hi + 1] <= t[lo] + window + 1e-9
    end = t[lo:hi + 1] >= t[hi] - window - 1e-9
    net_drop = float(np.median(initial_volume[begin]) - np.median(initial_volume[end]))
    result['observed_net_volume_drop'] = net_drop
    if net_drop <= 0:
        return fail('observed_volume_increase' if net_drop < -epsilon else 'no_net_discharge',
                    'observed volume increases rather than discharges' if net_drop < -epsilon else
                    'no observed net positive discharge')
    start, onset = _observed_discharge_start(initial_volume, t[lo:hi + 1], window,
                                             float(cfg.get('min_flow_snr', 3.)))
    result['discharge_onset'] = onset
    for i in range(lo, lo + start):
        result['frame_selection_reasons'][i] = 'initial_waiting_period_before_observable_discharge'
    lo += start
    selected = []
    for i in range(lo, hi + 1):
        if not selected or t[i] - t[selected[-1]] >= window - 1e-9:
            selected.append(i)
    result['independent_fit_frames'] = selected
    result['fit_window_sec'] = [float(t[lo]), float(t[hi])]
    if len(selected) < cfg['min_independent_fit_points']:
        return fail('insufficient_independent_observations', 'too few independent volume observations')
    local = np.asarray(selected) - lo
    # Trailing frames after the last independent observation cannot influence
    # the loss; exclude them from exponential scaling as well.
    fitted_end = selected[-1]
    tt = t[lo:fitted_end + 1]
    hh = h[lo:fitted_end + 1]
    result['fit_window_sec'][1] = float(t[fitted_end])
    vv = v[selected]
    vscale = float(np.ptp(vv))
    if vscale <= epsilon:
        return fail('no_volume_change', 'no observed volume change in discharge interval')
    hscale = float(np.median(hh))
    yn = (vv - vv.mean()) / vscale
    loghn = np.log(hh / hscale)
    if np.ptp(loghn) <= 1e-8:
        return fail('constant_head', 'constant observed head: exponent is unidentifiable')

    def evaluate(beta, details=False):
        # Removing a common log scale cannot change the fitted exponent. It
        # prevents overflow for large free beta or changes in length units.
        exponent = beta * loghn
        shift = float(np.max(exponent))
        weights = np.exp(exponent - shift)
        raw_integral = cumulative_trapezoid(weights, tt, initial=0)[local]
        scale = float(raw_integral[-1])
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            return (np.inf, None, None, None, weights, shift, scale) if details else np.inf
        integral = raw_integral / scale
        design = np.c_[np.ones(len(local)), integral]
        coef = np.linalg.lstsq(design, yn, rcond=None)[0]
        residual = yn - design @ coef
        loss = float(np.mean(residual ** 2)) if coef[1] < 0 else np.inf
        if details:
            return loss, coef, integral, residual, weights, shift, scale
        return loss

    bound = float(cfg.get('exponent_search_initial_bound', 8.))
    maximum = float(cfg.get('exponent_search_max_bound', 1024.))
    if not (np.isfinite(bound) and np.isfinite(maximum) and 0 < bound <= maximum):
        return fail('invalid_search_bounds', 'invalid numerical exponent search bounds')
    attempts = []
    while True:
        grid = np.linspace(-bound, bound, 129)
        values = np.asarray([evaluate(x) for x in grid])
        index = int(np.argmin(values))
        attempts.append({'bound': bound, 'best_grid_exponent': float(grid[index]),
                         'loss': float(values[index]) if np.isfinite(values[index]) else None})
        result['exponent_search'] = {'attempts': attempts, 'final_bounds': [-bound, bound],
                                     'target_exponent_used': False}
        if not np.isfinite(values[index]):
            return fail('no_positive_discharge_fit', 'no positive-discharge coefficient fits observed volume')
        if index not in (0, len(grid) - 1):
            break
        if bound >= maximum:
            return fail('exponent_search_exhausted',
                        'exponent optimum remains outside the expanded numerical search range')
        bound = min(2 * bound, maximum)
    fitted = minimize_scalar(evaluate, bounds=(grid[index - 1], grid[index + 1]),
                             method='bounded', options={'xatol': 1e-7})
    if not fitted.success or not np.isfinite(fitted.fun):
        return fail('optimizer_failed', 'integrated exponent optimization failed')
    beta = float(fitted.x)
    loss, coef, integral, residual, weights, shift, scale = evaluate(beta, True)
    raw_derivative = cumulative_trapezoid(weights * loghn, tt, initial=0)[local]
    derivative = (raw_derivative - integral * raw_derivative[-1]) / scale
    jacobian = np.c_[np.ones(len(local)), integral, coef[1] * derivative]
    condition = float(np.linalg.cond(jacobian))
    result['fit_condition_number'] = condition
    if not np.isfinite(condition) or condition > 1e10:
        return fail('ill_conditioned_exponent', 'integrated exponent fit is ill-conditioned')
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * float(np.sum(residual ** 2)) / max(1, len(local) - 3)
    se = float(np.sqrt(max(0, covariance[2, 2])))
    logmean = float(np.mean(np.log(h[selected])))
    logk = float(np.log(-coef[1]) + np.log(vscale) - np.log(scale) - shift - beta * np.log(hscale))
    # k is unit dependent and may exceed float range even when beta is sound.
    # Its logarithm remains exact; never fail or emit Infinity just for k.
    k = float(np.exp(logk)) if np.log(np.finfo(float).tiny) <= logk <= np.log(np.finfo(float).max) else None
    result['fit'] = {
        'velocity': beta, 'slope_se': se, 'slope_ci95': [beta - 1.96 * se, beta + 1.96 * se],
        'intercept_at_mean_time': float(logk + beta * logmean), 'log_height_mean': logmean,
        'independent_points': len(local), 'integrated_volume_residual_rms': float(np.sqrt(loss) * vscale),
        'normalised_volume_residual_rms': float(np.sqrt(loss)), 'positive_flow_coefficient': k,
        'log_positive_flow_coefficient': logk,
        'method': 'Integral conservation: V(t)=a-k*integral(h(t)**beta dt), unconstrained target exponent, stable adaptive numerical search from observed volume/head',
    }
    if k is None:
        result['warnings'].append('flow_coefficient_outside_float_range: logarithm reported')
    if 3.92 * se > 1:
        result['warnings'].append('wide_beta_interval: limited exponent precision')
    return result

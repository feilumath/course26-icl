#!/usr/bin/env python3
"""Self-consistent identity-attention ODEs; Python 3 standard library only.

Run: python3 dynamics.py
Outputs: results/dynamics.{csv,json,tex,svg,html}.

No centering, projection, damping, noise, clipping, or fixed-measure surrogate is
used. The normalized drift uses a shifted softmax. The unnormalized drift is
(1/N) sum_j exp(<x_i,x_j>/tau) x_j. All RK stages update every particle.
The calculation stops at max_i |x_i| = RADIUS_STOP; this is a numerical domain
boundary, not a modified ODE or a claim that the exact solution stops there.
"""

import argparse
import csv
import json
import math
from pathlib import Path


INITIAL = [(-0.80, -0.12), (-0.65, 0.22), (-0.45, -0.24), (-0.30, 0.12),
           (0.30, -0.12), (0.45, 0.24), (0.65, -0.22), (0.80, 0.12)]
TAUS = (0.5, 1.0, 2.0)
HORIZON = 1.6
RADIUS_STOP = 2.2
SAMPLES = 161
DISPLAY_TOLERANCE = 5e-5
COLORS = ('#0072B2', '#00856A', '#C86422')
TIKZ_COLORS = ('ipsBlue', 'ipsTeal', 'ipsOrange')


def radius(x):
    return max(math.hypot(x[i], x[i + 1]) for i in range(0, len(x), 2))


def drift(x, tau, model):
    """The current whole configuration enters the drift at every call."""
    n = len(x) // 2
    result = []
    for i in range(n):
        scores = [(x[2*i]*x[2*j] + x[2*i+1]*x[2*j+1])/tau for j in range(n)]
        offset = max(scores) if model == 'normalized' else 0.0
        weights = [math.exp(s - offset) for s in scores]
        denominator = sum(weights) if model == 'normalized' else n
        result.extend(sum(weights[j]*x[2*j+k] for j in range(n))/denominator
                      for k in (0, 1))
    return result


def rk4(x, h, tau, model):
    k1 = drift(x, tau, model)
    y2 = [a + h*b/2 for a, b in zip(x, k1)]
    if radius(y2) > 1.25*RADIUS_STOP:
        raise ArithmeticError('Trial stage beyond the numerical domain')
    k2 = drift(y2, tau, model)
    y3 = [a + h*b/2 for a, b in zip(x, k2)]
    if radius(y3) > 1.25*RADIUS_STOP:
        raise ArithmeticError('Trial stage beyond the numerical domain')
    k3 = drift(y3, tau, model)
    y4 = [a + h*b for a, b in zip(x, k3)]
    if radius(y4) > 1.25*RADIUS_STOP:
        raise ArithmeticError('Trial stage beyond the numerical domain')
    k4 = drift(y4, tau, model)
    return [a + h*(b + 2*c + 2*d + e)/6
            for a, b, c, d, e in zip(x, k1, k2, k3, k4)]


def two_half_steps(x, h, tau, model):
    return rk4(rk4(x, h/2, tau, model), h/2, tau, model)


def diagnostics(x, tau):
    n = len(x)//2
    mean = [sum(x[k::2])/n for k in (0, 1)]
    second = sum(z*z for z in x)/n
    potential = tau/(2*n*n)*sum(
        math.exp((x[2*i]*x[2*j] + x[2*i+1]*x[2*j+1])/tau)
        for i in range(n) for j in range(n))
    return {'max_radius': radius(x), 'rms_radius': math.sqrt(second),
            'variance': second - sum(m*m for m in mean),
            'energy': -potential, 'mean_x': mean[0], 'mean_y': mean[1]}


def snapshot(t, x, tau):
    return dict(time=t, particles=[[x[i], x[i+1]] for i in range(0, len(x), 2)],
                **diagnostics(x, tau))


def append_display_segment(history, t0, x0, t1, x1, tau, model, depth=0):
    """Save dense states with verified linear interpolation between them.

    Accepted RK steps resolve local integration error, but this alone does not
    control straight-line interpolation. Compare each candidate segment at its
    quarter, half, and three-quarter times with further RK4 half steps. Bisect
    whenever the maximum Euclidean error of any particle exceeds the display
    tolerance. The returned maximum concerns the retained segments only.
    """
    h = t1-t0
    references = {}
    errors = []
    for fraction in (0.25, 0.5, 0.75):
        reference = two_half_steps(x0, h*fraction, tau, model)
        references[fraction] = reference
        linear = [a + fraction*(b-a) for a, b in zip(x0, x1)]
        errors.append(max(math.hypot(reference[i]-linear[i], reference[i+1]-linear[i+1])
                          for i in range(0, len(x0), 2)))
    error = max(errors)
    if error > DISPLAY_TOLERANCE:
        if depth > 24:
            raise ArithmeticError('Display interpolation refinement did not converge')
        mid = references[0.5]
        left = append_display_segment(history, t0, x0, t0+h/2, mid, tau, model, depth+1)
        right = append_display_segment(history, t0+h/2, mid, t1, x1, tau, model, depth+1)
        return max(left, right)
    history.append(snapshot(t1, x1, tau))
    return error


def integrate(tau, model, tolerance=2e-10, max_step=0.015,
              initial=INITIAL, horizon=HORIZON, samples=SAMPLES):
    """Adaptive RK4 step-doubling, with a resolved radius-boundary event.

    The accepted state is the result of two half steps; the local-error estimate
    is their difference from one full step divided by 15. Trial states beyond a
    larger guard radius are rejected, never clipped. The first crossing of the
    requested radius boundary is localized by bisection of the accepted step.
    """
    x = [z for point in initial for z in point]
    assert radius(x) < RADIUS_STOP
    t, h, accepted, rejected = 0.0, max_step, 0, 0
    history = [snapshot(t, x, tau)]
    display_history = [snapshot(t, x, tau)]
    display_error = 0.0
    cutoff = False
    for target in [horizon*i/(samples-1) for i in range(1, samples)]:
        while t < target - 2e-15:
            step = min(h, target-t, max_step)
            if step < 1e-14:
                raise ArithmeticError('Adaptive step too small')
            try:
                full = rk4(x, step, tau, model)
                refined = two_half_steps(x, step, tau, model)
                error = max(abs(a-b) for a, b in zip(full, refined))/15
            except (ArithmeticError, OverflowError):
                h = step/2
                rejected += 1
                continue
            scale = tolerance*(1 + max(abs(a) for a in refined))
            if error > scale:
                h = step*max(0.1, 0.8*(scale/error)**0.2)
                rejected += 1
                continue
            accepted += 1
            if radius(refined) >= RADIUS_STOP:
                lo, hi = 0.0, step
                for _ in range(48):
                    mid = (lo+hi)/2
                    if radius(two_half_steps(x, mid, tau, model)) >= RADIUS_STOP:
                        hi = mid
                    else:
                        lo = mid
                final = two_half_steps(x, lo, tau, model)
                display_error = max(display_error, append_display_segment(
                    display_history, t, x, t+lo, final, tau, model))
                x, t = final, t+lo
                history.append(snapshot(t, x, tau))
                cutoff = True
                break
            display_error = max(display_error, append_display_segment(
                display_history, t, x, t+step, refined, tau, model))
            x, t = refined, t+step
            h = min(max_step, step*min(2.0, 0.9*(scale/max(error, 1e-30))**0.2))
        if cutoff:
            break
        history.append(snapshot(target, x, tau))
    return {'tau': tau, 'model': model, 'cutoff': cutoff,
            'status': 'radius boundary reached' if cutoff else 'time horizon reached',
            'stop_time': history[-1]['time'], 'accepted_steps': accepted,
            'rejected_steps': rejected, 'tolerance': tolerance, 'max_step': max_step,
            'history': history, 'display_history': display_history,
            'display_interpolation_max_checked_particle_error': display_error}


def self_checks(runs):
    x = [z for p in INITIAL for z in p]
    n = len(INITIAL)
    eps = 1e-6
    finite_difference_error = 0.0
    energy_identity_error = 0.0
    for tau in TAUS:
        raw = drift(x, tau, 'unnormalized')
        grad = []
        for k in range(len(x)):
            plus, minus = list(x), list(x)
            plus[k] += eps
            minus[k] -= eps
            grad.append((diagnostics(plus, tau)['energy'] -
                         diagnostics(minus, tau)['energy'])/(2*eps))
        finite_difference_error = max(finite_difference_error,
                                      max(abs(g + b/n) for g, b in zip(grad, raw)))
        for model in ('normalized', 'unnormalized'):
            velocity = drift(x, tau, model)
            derivative = sum(g*v for g, v in zip(grad, velocity))
            expected = -sum(a*b for a, b in zip(raw, velocity))/n
            energy_identity_error = max(energy_identity_error, abs(derivative-expected))
            assert derivative < 0
    assert finite_difference_error < 2e-9
    assert energy_identity_error < 2e-9
    one = integrate(0.5, 'normalized', initial=[(0.3, -0.2)], horizon=0.8, samples=81)
    one_error = max(abs(z - z0*math.exp(one['stop_time']))
                    for z, z0 in zip(one['history'][-1]['particles'][0], (0.3, -0.2)))
    assert one_error < 2e-9
    # Independent exact scalar unnormalized trajectory: t = integral from r0
    # to r of exp(-u^2/tau)/u du. Composite Simpson quadrature supplies a check
    # that does not reuse the ODE stepping formula.
    scalar = integrate(0.5, 'unnormalized', initial=[(0.5, 0)], horizon=1.6)
    r0, r1, tau = 0.5, scalar['history'][-1]['max_radius'], 0.5
    count = 20000
    step = (r1-r0)/count
    terms = [math.exp(-(r0+i*step)**2/tau)/(r0+i*step) for i in range(count+1)]
    scalar_time = step/3*(terms[0] + terms[-1] +
                         4*sum(terms[1:-1:2]) + 2*sum(terms[2:-1:2]))
    scalar_error = abs(scalar_time - scalar['stop_time'])
    assert scalar['cutoff'] and scalar_error < 2e-8
    convergence = []
    independent_display_checks = []
    for run in runs:
        fine = integrate(run['tau'], run['model'], tolerance=1e-11, max_step=0.0075)
        # Match fixed output times; the final, model-specific boundary event is
        # compared separately because its time is not a fixed-grid sample.
        worst = 0.0
        for coarse_sample, fine_sample in zip(run['history'], fine['history']):
            if abs(coarse_sample['time'] - fine_sample['time']) > 1e-12:
                break
            worst = max(worst, max(math.dist(a, b) for a, b in
                                   zip(coarse_sample['particles'], fine_sample['particles'])))
        time_error = abs(run['stop_time'] - fine['stop_time'])
        assert run['cutoff'] == fine['cutoff']
        assert worst < 2e-6 and time_error < 2e-7
        assert run['display_interpolation_max_checked_particle_error'] <= DISPLAY_TOLERANCE
        assert run['display_history'][0] == run['history'][0]
        assert run['display_history'][-1]['particles'] == run['history'][-1]['particles']
        values = [frame['energy'] for frame in run['history']]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
        assert all(abs(frame['mean_x']) + abs(frame['mean_y']) < 1e-10
                   for frame in run['history'])
        if run['model'] == 'unnormalized' and run['cutoff']:
            # Regression check for the fastest final interval: integrating
            # afresh to its midpoint catches coarse fixed-grid interpolation,
            # even when all stored sample positions are individually accurate.
            target = (run['history'][-2]['time'] + run['stop_time'])/2
            dense = run['display_history']
            index = next(i for i in range(len(dense)-1)
                         if dense[i]['time'] <= target <= dense[i+1]['time'])
            a, b = dense[index:index+2]
            fraction = (target-a['time'])/(b['time']-a['time'])
            interpolated = [[x + fraction*(y-x) for x, y in zip(p, q)]
                            for p, q in zip(a['particles'], b['particles'])]
            direct = integrate(run['tau'], run['model'], tolerance=1e-12,
                               max_step=0.00375, horizon=target, samples=2)
            error = max(math.dist(p, q) for p, q in
                        zip(interpolated, direct['history'][-1]['particles']))
            assert error < 1e-4
            independent_display_checks.append({'model': run['model'], 'tau': run['tau'],
                                               'time': target, 'max_particle_error': error})
        convergence.append({'tau': run['tau'], 'model': run['model'],
                            'max_particle_difference_at_shared_times': worst,
                            'cutoff_time_difference': time_error})
    return {'finite_difference_energy_gradient_max_error': finite_difference_error,
            'energy_dissipation_identity_max_error': energy_identity_error,
            'single_particle_normalized_exact_error': one_error,
            'single_particle_unnormalized_quadrature_time_error': scalar_error,
            'display_interpolation_max_checked_particle_error': max(
                run['display_interpolation_max_checked_particle_error'] for run in runs),
            'display_interpolation_check_fractions': [0.25, 0.5, 0.75],
            'display_interpolation_tolerance': DISPLAY_TOLERANCE,
            'independent_display_midpoint_checks': independent_display_checks,
            'refinement': convergence, 'passed': True}


def write_csv(path, runs):
    columns = ['model', 'tau', 'time', 'particle', 'x', 'y', 'max_radius',
               'rms_radius', 'variance', 'energy', 'mean_x', 'mean_y', 'is_cutoff']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for i, frame in enumerate(run['history']):
                for particle, point in enumerate(frame['particles']):
                    row = {key: frame[key] for key in columns if key in frame}
                    row.update(model=run['model'], tau=run['tau'], particle=particle,
                               x=point[0], y=point[1],
                               is_cutoff=run['cutoff'] and i == len(run['history'])-1)
                    writer.writerow(row)


def downsample(history, maximum=100):
    stride = max(1, len(history)//maximum)
    selected = history[::stride]
    if selected[-1] is not history[-1]:
        selected.append(history[-1])
    return selected


def write_tikz(path, runs):
    """Four-panel native TikZ, under 15 cm wide, no pgfplots requirement."""
    lines = [r'\begin{tikzpicture}[x=1cm,y=1cm,font=\scriptsize,text=ipsInk,',
             r'  line cap=round,line join=round]',
             r'\path[use as bounding box] (-.7,-.45) rectangle (13.65,6.65);']
    chosen = [run for run in runs if run['tau'] == 1.0]
    for panel, run in enumerate(chosen):
        left, bottom, size = panel*7.0, 3.9, 2.45
        cx, cy = left+3.0, bottom+0.95
        scale = 0.95
        lines.append(r'\begin{scope}')
        lines.append(fr'\node[font=\small\bfseries] at ({cx:.3f},6.35) {{{run["model"].capitalize()}, $\tau=1$}};')
        lines.append(fr'\draw[ipsGuide,->] ({left+.45:.3f},{cy:.3f}) -- ({left+5.55:.3f},{cy:.3f}) node[right,text=ipsInk] {{$x_1$}};')
        lines.append(fr'\draw[ipsGuide,->] ({cx:.3f},{bottom+.15:.3f}) -- ({cx:.3f},{bottom+2.05:.3f}) node[right,text=ipsInk] {{$x_2$}};')
        for i in range(len(INITIAL)):
            points = [(cx+scale*f['particles'][i][0], cy+scale*f['particles'][i][1])
                      for f in run['display_history']]
            color = 'ipsTeal' if i < 4 else 'ipsOrange'
            coords = ' '.join(f'({x:.4f},{y:.4f})' for x, y in points)
            lines.append(fr'\draw[{color},thick] plot coordinates {{{coords}}};')
            start, finish = points[0], points[-1]
            lines.append(fr'\draw[{color},fill=white] ({start[0]:.4f},{start[1]:.4f}) circle (1.3pt);')
            lines.append(fr'\fill[{color}] ({finish[0]:.4f},{finish[1]:.4f}) circle (1.7pt);')
        label = (fr'stopped at $R={RADIUS_STOP}$, $t={run["stop_time"]:.3f}$'
                 if run['cutoff'] else fr'$0\leq t\leq {run["stop_time"]:.1f}$')
        lines.append(fr'\node[align=center] at ({cx:.3f},3.72) {{{label}}};')
        lines.append(r'\end{scope}')
    # One fixed vertical scale in both radius panels; solid normalized and
    # dashed unnormalized, temperatures additionally encoded by color.
    for panel, model in enumerate(('normalized', 'unnormalized')):
        left, bottom, width, height = panel*7.0+0.25, 0.30, 5.55, 2.5
        lines.append(fr'\draw[ipsGuide,->] ({left:.3f},{bottom:.3f}) -- ({left+width+.2:.3f},{bottom:.3f}) node[right,text=ipsInk] {{$t$}};')
        lines.append(fr'\draw[ipsGuide,->] ({left:.3f},{bottom:.3f}) -- ({left:.3f},{bottom+height+.15:.3f}) node[above,text=ipsInk] {{$R(t)$}};')
        for tick in (0.0, 0.8, 1.6):
            pos = left + width*tick/HORIZON
            lines.append(fr'\draw[ipsGuide] ({pos:.3f},{bottom:.3f}) -- ++(0,-.05) node[below,text=ipsInk] {{{tick:g}}};')
        for tick in (1.0, 2.0):
            pos = bottom+height*tick/2.5
            lines.append(fr'\draw[ipsGuide] ({left:.3f},{pos:.3f}) -- ++(-.05,0) node[left,text=ipsInk] {{{tick:g}}};')
        lines.append(fr'\draw[ipsGuide,densely dotted] ({left:.3f},{bottom+height*RADIUS_STOP/2.5:.3f}) -- ++({width:.3f},0);')
        for k, tau in enumerate(TAUS):
            run = next(r for r in runs if r['model'] == model and r['tau'] == tau)
            pts = [(left+width*f['time']/HORIZON, bottom+height*f['max_radius']/2.5)
                   for f in run['display_history']]
            coords = ' '.join(f'({x:.4f},{y:.4f})' for x, y in pts)
            style = 'thick' if model == 'normalized' else 'thick,dashed'
            lines.append(fr'\draw[{TIKZ_COLORS[k]},{style}] plot coordinates {{{coords}}};')
            end = pts[-1]
            lines.append(fr'\fill[{TIKZ_COLORS[k]}] ({end[0]:.4f},{end[1]:.4f}) circle (1.6pt);')
            lx = left+0.7+k*1.7
            lines.append(fr'\draw[{TIKZ_COLORS[k]},{style}] ({lx:.3f},-.34) -- ++(.28,0) node[right,text=ipsInk] {{$\tau={tau:g}$}};')
    lines.append(r'\end{tikzpicture}')
    path.write_text('\n'.join(lines)+'\n')


def write_svg(path, runs):
    width, height = 1060, 730
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text {font-family:Arial,sans-serif;fill:#263238} .axis {stroke:#AAB2BC;stroke-width:1}</style>',
              '<text x="530" y="25" text-anchor="middle" font-size="21">Identity attention: finite-horizon particle growth</text>']
    for col, model in enumerate(('normalized', 'unnormalized')):
        run = next(r for r in runs if r['tau'] == 1 and r['model'] == model)
        cx, cy, scale = 265+530*col, 204, 62
        chunks.append(f'<text x="{cx}" y="58" text-anchor="middle" font-size="18">{model.capitalize()}, τ = 1</text>')
        chunks += [f'<line class="axis" x1="{cx-190}" y1="{cy}" x2="{cx+190}" y2="{cy}"/>',
                   f'<line class="axis" x1="{cx}" y1="{cy-120}" x2="{cx}" y2="{cy+120}"/>']
        for i in range(len(INITIAL)):
            color = '#00856A' if i < 4 else '#C86422'
            points = [(cx+scale*f['particles'][i][0], cy-scale*f['particles'][i][1]) for f in run['display_history']]
            poly = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
            chunks.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
            for pos, fill, radius_ in [(points[0], 'white', 3), (points[-1], color, 4)]:
                chunks.append(f'<circle cx="{pos[0]:.3f}" cy="{pos[1]:.3f}" r="{radius_}" fill="{fill}" stroke="{color}"/>')
        status = f'Stopped at radius {RADIUS_STOP}, t = {run["stop_time"]:.4f}' if run['cutoff'] else f'Time horizon t = {HORIZON}'
        chunks.append(f'<text x="{cx}" y="344" text-anchor="middle" font-size="14">{status}; open = initial, filled = final</text>')
        left, top, pw, ph = 65+530*col, 410, 425, 230
        chunks.append(f'<text x="{cx}" y="382" text-anchor="middle" font-size="17">Maximum particle radius R(t)</text>')
        for value in (0, 1, 2):
            y = top+ph*(1-value/2.5)
            chunks.append(f'<line class="axis" x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}"/><text x="{left-12}" y="{y+4}" text-anchor="end" font-size="12">{value}</text>')
        for value in (0, 0.4, 0.8, 1.2, 1.6):
            x = left+pw*value/HORIZON
            chunks.append(f'<text x="{x}" y="{top+ph+20}" text-anchor="middle" font-size="12">{value:g}</text>')
        for k, tau in enumerate(TAUS):
            run = next(r for r in runs if r['model'] == model and r['tau'] == tau)
            points = [(left+pw*f['time']/HORIZON, top+ph*(1-f['max_radius']/2.5)) for f in run['display_history']]
            poly = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
            dashed = ' stroke-dasharray="6 3"' if model == 'unnormalized' else ''
            chunks.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[k]}" stroke-width="2.2"{dashed}/>')
            end = points[-1]
            chunks.append(f'<circle cx="{end[0]}" cy="{end[1]}" r="3.4" fill="{COLORS[k]}"/>')
            chunks.append(f'<text x="{left+35+135*k}" y="688" font-size="14" style="fill:{COLORS[k]}">τ = {tau:g}</text>')
        boundary = top+ph*(1-RADIUS_STOP/2.5)
        chunks.append(f'<line x1="{left}" y1="{boundary}" x2="{left+pw}" y2="{boundary}" stroke="#AAB2BC" stroke-dasharray="2 5"/>')
    chunks.append('<text x="530" y="720" text-anchor="middle" font-size="13">All curves stop at their actual final time; the radius boundary is a numerical stopping rule.</text></svg>')
    path.write_text('\n'.join(chunks)+'\n')


HTML_TEMPLATE = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-consistent identity attention</title>
<style>
body{font:16px system-ui,sans-serif;color:#263238;margin:24px auto;padding:0 20px;max-width:1200px;background:#f7f8fa}
h1{font-size:25px}p{max-width:1000px;line-height:1.5}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:white;padding:14px;border-radius:8px}
button,select{font:inherit;padding:6px 10px}input[type=range]{width:min(430px,70vw)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.panel{background:white;padding:14px;border-radius:8px}canvas{width:100%;height:auto;border:1px solid #e7e9ec}.status{min-height:55px;line-height:1.4;font-size:14px}.legend{font-size:14px}.stop{color:#94491b;font-weight:600}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;text-align:right;border-bottom:1px solid #eee}td:first-child,th:first-child{text-align:left}summary{cursor:pointer}@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style>
<h1>Self-consistent identity attention</h1>
<p>Eight particles evolve with Q = K = V = I. The normalized drift divides the weighted sum by the sum of weights; the unnormalized drift divides by N. Every drift evaluation updates the empirical measure. This symmetric cloud spreads while its interaction energy decreases: gradient descent alone does not imply clustering.</p>
<div class="controls"><button id="play">Play</button><button id="reset">Reset</button><label>Temperature τ <select id="tau"><option>0.5</option><option selected>1</option><option>2</option></select></label><label>Time <input id="time" type="range" min="0" max="1.6" step="any" value="0"></label><output id="clock">0.000</output><label><input id="field" type="checkbox" checked> field directions</label></div>
<div class="grid"><section class="panel"><h2>Normalized</h2><canvas id="normalized" width="550" height="400"></canvas><p id="normalized-status" class="status"></p><table id="normalized-table"></table></section><section class="panel"><h2>Unnormalized</h2><canvas id="unnormalized" width="550" height="400"></canvas><p id="unnormalized-status" class="status"></p><table id="unnormalized-table"></table></section></div>
<p class="legend">Open circles: initial states. Filled circles: current states. Colored lines: recorded trajectories. Gray arrows: directions of the current attention field (unit length for readability; they do not display speed). The same spatial scale is used in both panels.</p>
<p><strong>Numerical stopping rule.</strong> Each computation stops when the largest particle radius reaches 2.2, or at t = 1.6. A stopped panel explicitly displays its own final time and no field arrows; it does not represent a stationary continuation. No drift clipping or state projection is used. Displayed positions are linearly interpolated between saved adaptive states, further refined until quarter-, half-, and three-quarter-time checks against refined RK4 steps are within 5×10⁻⁵ per particle.</p>
<details><summary>Equations and reproducibility</summary><p>Normalized: dxᵢ/dt = Σⱼ exp(xᵢ·xⱼ/τ)xⱼ / Σⱼ exp(xᵢ·xⱼ/τ). Unnormalized: dxᵢ/dt = (1/N)Σⱼ exp(xᵢ·xⱼ/τ)xⱼ. Energy: F = −τ/(2N²) Σᵢⱼ exp(xᵢ·xⱼ/τ). Variance: (1/N)Σᵢ|xᵢ−mean(x)|².</p><p>Run <code>python3 dynamics.py</code> in the parent folder. Adaptive RK4 with step doubling, tolerance 2×10⁻¹⁰ and maximum step 0.015. Refinement checks use tolerance 10⁻¹¹ and half the maximum step. The script also checks the finite-particle energy gradient, the dissipation identity, exact one-particle behavior, and display interpolation. CSV diagnostics use the fixed output grid; JSON includes both fixed-grid and adaptive display histories. All data and code used by this page are embedded; no internet connection is required.</p></details>
<script>
const DATA = __DATA__;
const slider=document.getElementById('time'), select=document.getElementById('tau');
let running=false, previous=null;
const colors=['#00856A','#00856A','#00856A','#00856A','#C86422','#C86422','#C86422','#C86422'];
function frameAt(run,t){let h=run.display_history;if(t>=run.stop_time)return h[h.length-1];let k=0;while(k+1<h.length&&h[k+1].time<t)k++;let a=h[k],b=h[Math.min(k+1,h.length-1)],q=b.time===a.time?0:(t-a.time)/(b.time-a.time);return {...a,time:t,particles:a.particles.map((p,i)=>p.map((z,j)=>z+q*(b.particles[i][j]-z)))};}
function statistics(points,tau){let n=points.length,m=[0,0],s=0,z=0;for(let p of points){m[0]+=p[0]/n;m[1]+=p[1]/n;s+=p[0]*p[0]+p[1]*p[1];for(let q of points)z+=Math.exp((p[0]*q[0]+p[1]*q[1])/tau)}return{radius:Math.max(...points.map(p=>Math.hypot(...p))),variance:s/n-m[0]*m[0]-m[1]*m[1],energy:-tau*z/(2*n*n)}}
function direction(x,points,tau){let scores=points.map(p=>(x[0]*p[0]+x[1]*p[1])/tau),max=Math.max(...scores),weights=scores.map(s=>Math.exp(s-max));let v=[0,0];points.forEach((p,i)=>{v[0]+=weights[i]*p[0];v[1]+=weights[i]*p[1]});let n=Math.hypot(...v);return n>1e-10?v.map(z=>z/n):[0,0]}
function render(){let time=+slider.value,tau=+select.value;document.getElementById('clock').textContent=time.toFixed(3);for(let model of ['normalized','unnormalized']){let run=DATA.runs.find(r=>r.model===model&&r.tau===tau),done=run.cutoff&&time>=run.stop_time-1e-12,frame=frameAt(run,time),canvas=document.getElementById(model),c=canvas.getContext('2d');let px=x=>275+75*x,py=y=>200-75*y;c.clearRect(0,0,550,400);c.strokeStyle='#c7cdd3';c.lineWidth=1;c.beginPath();c.moveTo(30,200);c.lineTo(520,200);c.moveTo(275,20);c.lineTo(275,380);c.stroke();c.fillStyle='#67727a';c.font='12px system-ui';for(let tick of [-2,-1,1,2]){c.fillText(tick,px(tick)-5,217);c.fillText(tick,279,py(tick)+4)}c.fillText('x₁',525,204);c.fillText('x₂',279,16);if(document.getElementById('field').checked&&!done){c.strokeStyle='#bec6cc';c.lineWidth=1;for(let x=-2.5;x<=2.51;x+=.5)for(let y=-2;y<=2.01;y+=.5){let v=direction([x,y],frame.particles,tau);if(Math.hypot(...v)<.1)continue;let a=[px(x),py(y)],b=[a[0]+10*v[0],a[1]-10*v[1]],angle=Math.atan2(b[1]-a[1],b[0]-a[0]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.moveTo(b[0]-4*Math.cos(angle-.5),b[1]-4*Math.sin(angle-.5));c.lineTo(...b);c.lineTo(b[0]-4*Math.cos(angle+.5),b[1]-4*Math.sin(angle+.5));c.stroke()}}
for(let i=0;i<frame.particles.length;i++){c.strokeStyle=colors[i];c.lineWidth=2;c.beginPath();let first=true;for(let f of run.display_history){if(f.time>Math.min(time,run.stop_time))break;let p=f.particles[i];if(first)c.moveTo(px(p[0]),py(p[1]));else c.lineTo(px(p[0]),py(p[1]));first=false}let p=frame.particles[i];c.lineTo(px(p[0]),py(p[1]));c.stroke();let initial=run.history[0].particles[i];c.beginPath();c.arc(px(initial[0]),py(initial[1]),4,0,2*Math.PI);c.fillStyle='white';c.fill();c.stroke();c.beginPath();c.arc(px(p[0]),py(p[1]),5,0,2*Math.PI);c.fillStyle=colors[i];c.fill()}
let status=document.getElementById(model+'-status');status.className='status'+(done?' stop':'');status.textContent=done?'COMPUTATION STOPPED at t = '+run.stop_time.toFixed(5)+' (radius 2.2). Showing this final recorded state; no continuation is computed.':'Displayed time t = '+frame.time.toFixed(3)+'. '+(run.cutoff?'This run reaches the radius boundary at t = '+run.stop_time.toFixed(5)+'.':'This run reaches the time horizon t = 1.6.');let s=statistics(frame.particles,tau);document.getElementById(model+'-table').innerHTML='<tr><th>Diagnostic</th><th>Current value</th></tr><tr><td>Maximum radius</td><td>'+s.radius.toFixed(4)+'</td></tr><tr><td>Variance</td><td>'+s.variance.toFixed(4)+'</td></tr><tr><td>Interaction energy F</td><td>'+s.energy.toFixed(5)+'</td></tr>';}}
document.getElementById('play').onclick=()=>{running=!running;if(+slider.value>=1.6)slider.value=0;document.getElementById('play').textContent=running?'Pause':'Play';previous=null};document.getElementById('reset').onclick=()=>{running=false;slider.value=0;document.getElementById('play').textContent='Play';render()};slider.oninput=render;select.onchange=()=>{slider.value=0;render()};document.getElementById('field').onchange=render;function tick(now){if(running&&previous!==null){slider.value=Math.min(1.6,+slider.value+(now-previous)*.00013);render();if(+slider.value>=1.6){running=false;document.getElementById('play').textContent='Play'}}previous=now;requestAnimationFrame(tick)}
window.attentionDemo={data:DATA,render:render,setTime(t){slider.value=Math.max(0,Math.min(1.6,t));render()},setTau(t){if(!DATA.temperatures.includes(t))throw Error('Invalid temperature');select.value=String(t);render()},getState(){return{time:+slider.value,tau:+select.value,playing:running,statuses:['normalized','unnormalized'].map(m=>document.getElementById(m+'-status').textContent)}}};render();requestAnimationFrame(tick);
</script></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [integrate(tau, model) for model in ('normalized', 'unnormalized') for tau in TAUS]
    checks = self_checks(runs)
    data = {'description': 'Raw self-consistent identity-attention ODEs on R^2',
            'initial_particles': INITIAL, 'temperatures': TAUS, 'horizon': HORIZON,
            'radius_stop': RADIUS_STOP,
            'stopping_rule': 'First max_i |x_i| = 2.2 or t = 1.6; no energy cutoff, drift clipping, or state projection',
            'energy_definition': '-tau/(2*N*N) sum_i sum_j exp(dot(x_i,x_j)/tau)',
            'history_description': 'history and CSV: fixed-grid output plus boundary event; display_history: accepted adaptive states with extra interpolation refinement',
            'display_interpolation_description': 'Linear interpolation checked at fractions .25, .5 and .75 against refined RK4 steps; per-particle Euclidean error at most 5e-5 at these check points',
            'checks': checks, 'runs': runs}
    (args.output/'dynamics.json').write_text(json.dumps(data, indent=2)+'\n')
    write_csv(args.output/'dynamics.csv', runs)
    write_tikz(args.output/'dynamics.tex', runs)
    write_svg(args.output/'dynamics.svg', runs)
    (args.output/'dynamics.html').write_text(HTML_TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    print(json.dumps({'checks': checks, 'runs': [dict(model=r['model'], tau=r['tau'],
                     cutoff=r['cutoff'], stop_time=r['stop_time'],
                     final={k:v for k,v in r['history'][-1].items() if k != 'particles'})
                     for r in runs]}, indent=2))


if __name__ == '__main__':
    main()

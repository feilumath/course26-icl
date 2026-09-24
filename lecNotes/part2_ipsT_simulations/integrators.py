#!/usr/bin/env python3
"""Reproduce the consensus integrator exercise using only Python's standard library.

The implicit equations are solved exactly for this linear benchmark. This is
not representative of the nonlinear solve cost for general attention fields.
"""

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = ("Euler", "Midpoint RK2", "Classical RK4", "Implicit Euler", "Implicit midpoint")
ORDERS = (1, 2, 4, 1, 2)
COLORS = ("#0072B2", "#00856A", "#C86422", "#7B61A8", "#263238")
INITIAL = (-2.0, -0.5, 0.25, 1.0, 3.0)


def field(x):
    mean = math.fsum(x) / len(x)
    return [mean - xi for xi in x]


def add(x, v, h):
    return [xi + h * vi for xi, vi in zip(x, v)]


def step(x, h, method):
    """Advance the entire cloud, rebuilding its mean at every explicit stage."""
    if method == "Euler":
        return add(x, field(x), h)
    if method == "Midpoint RK2":
        return add(x, field(add(x, field(x), h / 2)), h)
    if method == "Classical RK4":
        k1 = field(x)
        k2 = field(add(x, k1, h / 2))
        k3 = field(add(x, k2, h / 2))
        k4 = field(add(x, k3, h))
        return [xi + h * (a + 2 * b + 2 * c + d) / 6
                for xi, a, b, c, d in zip(x, k1, k2, k3, k4)]
    mean = math.fsum(x) / len(x)
    if method == "Implicit Euler":
        factor = 1 / (1 + h)
    elif method == "Implicit midpoint":
        factor = (1 - h / 2) / (1 + h / 2)
    else:
        raise ValueError(f"Unknown method: {method}")
    return [mean + factor * (xi - mean) for xi in x]


def amplification(h, method):
    return {
        "Euler": 1 - h,
        "Midpoint RK2": 1 - h + h * h / 2,
        "Classical RK4": 1 - h + h**2 / 2 - h**3 / 6 + h**4 / 24,
        "Implicit Euler": 1 / (1 + h),
        "Implicit midpoint": (1 - h / 2) / (1 + h / 2),
    }[method]


def rms(x):
    return math.sqrt(math.fsum(xi * xi for xi in x) / len(x))


def simulate(method, h, n):
    x = list(INITIAL)
    trajectory = [x]
    for _ in range(n):
        x = step(x, h, method)
        trajectory.append(x)
    return trajectory


def write_svg(path, rows, stability):
    """A transparent SVG plot; exact data are also written to CSV."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500">',
           '<rect width="1100" height="500" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#263238">',
           '<text x="550" y="25" text-anchor="middle" font-size="19">Consensus benchmark: accuracy and damping</text>']
    panels = [(75, 60, 425, 330), (650, 60, 390, 330)]
    for panel, title in zip(panels, ["Error at T = 2", "Centered-cloud size, h = 2.5"]):
        x, y, w, h = panel
        svg.append(f'<text x="{x + w/2}" y="{y-12}" text-anchor="middle">{title}</text>')
        svg.append(f'<path d="M{x},{y} V{y+h} H{x+w}" fill="none" stroke="#AAB2BC"/>')
    x, y, w, height = panels[0]
    for exponent in (-2, -4, -6, -8, -10, -12):
        py = y + height * (-exponent - 1) / 11
        svg.append(f'<text x="{x-9}" y="{py+5}" text-anchor="end">10^{exponent}</text>')
        svg.append(f'<path d="M{x},{py} H{x+w}" stroke="#e8ecef"/>')
    for n in (8, 16, 32, 64, 128):
        px = x + w * math.log2(n / 8) / 4
        svg.append(f'<text x="{px}" y="{y+height+22}" text-anchor="middle">{n}</text>')
    svg.append(f'<text x="{x+w/2}" y="{y+height+45}" text-anchor="middle">Number of steps (log scale)</text>')
    for method, color in zip(METHODS, COLORS):
        values = [r for r in rows if r['method'] == method]
        points = ' '.join(f"{x+w*math.log2(r['steps']/8)/4:.2f},{y+height*(-math.log10(r['rms_error'])-1)/11:.2f}" for r in values)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in values:
            px = x + w * math.log2(r['steps']/8) / 4
            py = y + height * (-math.log10(r['rms_error'])-1) / 11
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')
    x, y, w, height = panels[1]
    for exponent in (-18, -12, -6, 0, 4):
        py = y + height * (4 - exponent) / 22
        svg.append(f'<text x="{x-9}" y="{py+5}" text-anchor="end">10^{exponent}</text>')
        svg.append(f'<path d="M{x},{py} H{x+w}" stroke="#e8ecef"/>')
    for n in (0, 4, 8, 12, 16):
        svg.append(f'<text x="{x+w*n/16}" y="{y+height+22}" text-anchor="middle">{n}</text>')
    svg.append(f'<text x="{x+w/2}" y="{y+height+45}" text-anchor="middle">Step number; ratio to initial RMS spread</text>')
    for method, color in zip(METHODS, COLORS):
        factor = abs(amplification(2.5, method))
        points = ' '.join(f'{x+w*n/16:.2f},{y+height*(4-n*math.log10(factor))/22:.2f}' for n in range(17))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        x = 40 + i * 216
        svg.append(f'<path d="M{x},468 h24" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x+29}" y="473">{method}</text>')
    svg.append('</g></svg>')
    path.write_text('\n'.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mean = math.fsum(INITIAL) / len(INITIAL)
    exact = [mean + math.exp(-2) * (xi - mean) for xi in INITIAL]
    rows, checks = [], {}
    for method, order in zip(METHODS, ORDERS):
        errors = []
        for n in (8, 16, 32, 64, 128):
            h = 2 / n
            final = simulate(method, h, n)[-1]
            error = rms([a - b for a, b in zip(final, exact)])
            errors.append(error)
            rows.append(dict(method=method, steps=n, step_size=h, rms_error=error))
            if abs(math.fsum(final) / len(final) - mean) > 1e-12:
                raise AssertionError('Consensus mean is not preserved')
        observed = math.log2(errors[-2] / errors[-1])
        if abs(observed - order) > 0.15:
            raise AssertionError(f'{method}: expected order {order}, observed {observed}')
        # Compare an actual stage update to the independently derived scalar factor.
        for h in (0.5, 1.5, 2.5, 8.0):
            actual = step(INITIAL, h, method)
            target = [mean + amplification(h, method) * (xi-mean) for xi in INITIAL]
            if max(abs(a-b) for a, b in zip(actual, target)) > 1e-11:
                raise AssertionError(f'{method}: incorrect centered-mode amplification')
        checks[method] = dict(expected_order=order, observed_order=observed, mean_preserved=True)
    with (args.output / 'integrators_accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stability = [dict(method=method, step_size=h, factor=amplification(h, method),
                      contracts=abs(amplification(h, method)) < 1)
                 for method in METHODS for h in (0.5, 1.5, 2.5, 8.0)]
    with (args.output / 'integrators_stability.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stability[0]))
        writer.writeheader()
        writer.writerows(stability)
    summary = dict(model='dx_i/dt = mean(x) - x_i', initial=INITIAL, horizon=2,
                   checks=checks, implicit_solves='Exact for this linear system only')
    (args.output / 'integrators_checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_svg(args.output / 'integrators.svg', rows, stability)
    print(json.dumps(checks, indent=2))
    print(f'Wrote consensus benchmark results to {args.output}')


if __name__ == '__main__':
    main()

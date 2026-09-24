#!/usr/bin/env python3
"""Exact two-key attention sensitivity, using only the Python standard library.

Run from any directory:
    python3 sensitivity.py
    python3 sensitivity.py --radius 1 --temperatures .25 .5 1 2

Outputs are deterministic (no Monte Carlo noise).  The generated TikZ picture
uses the ipsInk/ipsGuide/ipsBlue/ipsTeal/ipsOrange/ipsPurple palette of the note.
All generated files have the prefix ``sensitivity``.  Open sensitivity.html
directly in a browser for an offline interactive version; no server is needed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


INK, GUIDE = "#263238", "#AAB2BC"
COLORS = ["#0072B2", "#00856A", "#C86422", "#7B61A8"]
TIKZ_COLORS = ["ipsBlue", "ipsTeal", "ipsOrange", "ipsPurple"]
SVG_DASHES = ["", "9 4", "2 4", "9 3 2 3"]
TIKZ_DASHES = ["solid", "dashed", "dotted", "dashdotted"]
LN10 = math.log(10)


def sigmoid(z: float) -> float:
    """Stable logistic function, including large positive/negative inputs."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def logaddexp(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def query_values(x: float, radius: float, tau: float) -> tuple[float, float]:
    z = radius * x / tau
    # sech(z)^2 = 4 exp(-2|z|)/(1+exp(-2|z|))^2 avoids overflow
    # and loss of precision from subtracting tanh(z)^2 from 1.
    e = math.exp(-2 * abs(z))
    derivative = (radius * radius / tau) * 4 * e / (1 + e) ** 2
    return radius * math.tanh(z), derivative


def rare_values(log_eps: float, radius: float, tau: float) -> dict[str, float]:
    """Use log epsilon and return log amplification without cancellation.

    Computing (Phi_eps-Phi_0)/(2R epsilon) directly loses precision when
    epsilon is tiny.  Instead use A=1/[epsilon+(1-epsilon)exp(-2R^2/tau)].
    """
    eps = math.exp(log_eps)
    scale = 2 * radius * radius / tau
    if log_eps >= 0:
        p, log_gain = 1.0, 0.0
    else:
        log_remaining = math.log(-math.expm1(log_eps))
        log_gain = -logaddexp(log_eps, log_remaining - scale)
        p = sigmoid(log_eps - log_remaining + scale)
    return {"epsilon": eps, "positive_key_weight": p,
            "field": radius * (2 * p - 1),
            "field_change": 2 * radius * p,
            "w1_change": 2 * radius * eps,
            "log10_amplification": log_gain / LN10}


def grid(lo: float, hi: float, count: int) -> list[float]:
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.3g}"


def panel_specs(radius: float, temperatures: list[float], emin: float):
    derivative_max = radius * radius / min(temperatures)
    log_gain_max = 2 * radius * radius / min(temperatures) / LN10
    return [
        {"title": "(a) Query field", "tex": r"(a) Query field $\Phi_\theta[\mu](x)$",
         "xlabel": "query x", "xtex": r"query $x$", "xmin": -radius,
         "xmax": radius, "ymin": -1.08 * radius, "ymax": 1.08 * radius,
         "xticks": [-radius, 0, radius], "yticks": [-radius, 0, radius]},
        {"title": "(b) Query derivative", "tex": r"(b) Query derivative $\partial_x\Phi_\theta[\mu](x)$",
         "xlabel": "query x", "xtex": r"query $x$", "xmin": -radius,
         "xmax": radius, "ymin": 0, "ymax": 1.08 * derivative_max,
         "xticks": [-radius, 0, radius],
         "yticks": [derivative_max * i / 4 for i in range(5)]},
        {"title": "(c) Field after adding rare mass", "tex": r"(c) Rare-mass field $\Phi_\theta[\mu_\varepsilon](R)$",
         "xlabel": "added mass ε (log scale)", "xtex": r"added mass $\varepsilon$ (log scale)",
         "xmin": emin, "xmax": 0, "ymin": -1.08 * radius,
         "ymax": 1.08 * radius, "xticks": [emin * (1 - i / 4) for i in range(5)],
         "yticks": [-radius, 0, radius], "logx": True},
        {"title": "(d) Field change / Wasserstein change", "tex": r"(d) Amplification $|\Delta\Phi|/W_1$",
         "xlabel": "added mass ε (log scale)", "xtex": r"added mass $\varepsilon$ (log scale)",
         "xmin": emin, "xmax": 0, "ymin": 0,
         "ymax": max(0.05, 1.08 * log_gain_max),
         "xticks": [emin * (1 - i / 4) for i in range(5)],
         "yticks": [i * max(1, math.ceil(log_gain_max / 4))
                    for i in range(math.floor(log_gain_max / max(1, math.ceil(log_gain_max / 4))) + 1)],
         "logx": True, "logy": True},
    ]


def build_data(radius, temperatures, points, emin):
    query_rows, rare_rows, curves = [], [], [[], [], [], []]
    for tau in temperatures:
        field_curve, derivative_curve, rare_curve, gain_curve = [], [], [], []
        for x in grid(-radius, radius, points):
            field, derivative = query_values(x, radius, tau)
            query_rows.append({"radius": radius, "temperature": tau, "query": x,
                               "field": field, "query_derivative": derivative})
            field_curve.append((x, field))
            derivative_curve.append((x, derivative))
        for log10_eps in grid(emin, 0, points):
            values = rare_values(log10_eps * LN10, radius, tau)
            rare_rows.append({"radius": radius, "temperature": tau,
                              "log10_epsilon": log10_eps, **values})
            rare_curve.append((log10_eps, values["field"]))
            gain_curve.append((log10_eps, values["log10_amplification"]))
        for target, curve in zip(curves, [field_curve, derivative_curve, rare_curve, gain_curve]):
            target.append(curve)
    return query_rows, rare_rows, curves


def verify(radius, temperatures):
    """Independent softmax/finite-difference checks, plus exact limiting bounds."""
    max_field_error, max_derivative_error, max_weight_error = 0.0, 0.0, 0.0
    for tau in temperatures:
        for x in grid(-radius, radius, 31):
            a, b = -radius * x / tau, radius * x / tau
            m = max(a, b)
            wa, wb = math.exp(a - m), math.exp(b - m)
            direct = radius * (wb - wa) / (wa + wb)
            field, derivative = query_values(x, radius, tau)
            max_field_error = max(max_field_error, abs(field - direct))
            h = min(radius, tau / radius) * 1e-5
            finite_difference = (query_values(x + h, radius, tau)[0]
                                 - query_values(x - h, radius, tau)[0]) / (2 * h)
            max_derivative_error = max(max_derivative_error, abs(derivative - finite_difference))
        for log_eps in [-1e-6, -.1, -1, -4, -10, -30, -100]:
            values = rare_values(log_eps, radius, tau)
            eps = math.exp(log_eps)
            a = math.log1p(-eps) - radius * radius / tau
            b = log_eps + radius * radius / tau
            m = max(a, b)
            direct_weight = math.exp(b - m) / (math.exp(a - m) + math.exp(b - m))
            max_weight_error = max(max_weight_error, abs(values["positive_key_weight"] - direct_weight))
            log_gain = values["log10_amplification"] * LN10
            assert -1e-12 <= log_gain <= min(2 * radius * radius / tau, -log_eps) + 1e-10
        scale = 2 * radius * radius / tau
        # epsilon*exp(scale) is <= exp(-35), so the ratio to the limiting
        # amplification must be almost one even when exp(scale) overflows.
        very_small = rare_values(-scale - 35, radius, tau)
        assert abs(very_small["log10_amplification"] * LN10 - scale) < 1e-10 * max(1, scale)
        _, derivative_zero = query_values(0, radius, tau)
        assert math.isclose(derivative_zero, radius * radius / tau, rel_tol=1e-14)
        half_mass_log = -logaddexp(0, scale)
        assert math.isclose(rare_values(half_mass_log, radius, tau)["positive_key_weight"], .5,
                            rel_tol=1e-12, abs_tol=1e-12)
    assert max_field_error <= 1e-12 * max(1, radius)
    assert max_weight_error <= 1e-12
    assert max_derivative_error <= 1e-6 * max(1, radius * radius / min(temperatures))
    return {"max_field_vs_weighted_softmax_error": max_field_error,
            "max_derivative_vs_central_difference_error": max_derivative_error,
            "max_rare_weight_vs_weighted_softmax_error": max_weight_error,
            "additional_checks": ["derivative maximum R^2/tau at x=0",
                                  "1 <= amplification <= min(exp(2R^2/tau), 1/epsilon)",
                                  "rare-mass amplification limit exp(2R^2/tau)",
                                  "half attention weight at epsilon=1/(1+exp(2R^2/tau))"]}


def tick_label(value, logarithmic, tex=False):
    if logarithmic:
        return (rf"$10^{{{fmt(value)}}}$" if tex else f"10^{fmt(value)}")
    return (f"${fmt(value)}$" if tex else fmt(value))


def write_tikz(path, panels, curves, temperatures, radius):
    width, height = 6.05, 2.65
    lines = ["% Generated by sensitivity.py; exact two-key formulas, no random sampling.",
             r"\begin{tikzpicture}[x=1cm,y=1cm,color=ipsInk,",
             r"  every node/.style={text=ipsInk,font=\scriptsize},line cap=round]",
             r"\path[use as bounding box] (-.68,-5.85) rectangle (13.92,3.17);"]
    for i, panel in enumerate(panels):
        ox, oy = (i % 2) * 7.65, -(i // 2) * 4.3
        def xy(x, y):
            return (ox + width * (x - panel["xmin"]) / (panel["xmax"] - panel["xmin"]),
                    oy + height * (y - panel["ymin"]) / (panel["ymax"] - panel["ymin"]))
        lines.append(rf"\node[font=\small] at ({ox + width/2:.4f},{oy+height+.30:.4f}) {{{panel['tex']}}};")
        for y in panel["yticks"]:
            _, py = xy(panel["xmin"], y)
            lines.append(rf"\draw[ipsGuide!42,thin] ({ox:.4f},{py:.4f}) -- ({ox+width:.4f},{py:.4f});")
            lines.append(rf"\node[anchor=east] at ({ox-.08:.4f},{py:.4f}) {{{tick_label(y,panel.get('logy'),True)}}};")
        for x in panel["xticks"]:
            px, _ = xy(x, panel["ymin"])
            lines.append(rf"\draw[ipsGuide] ({px:.4f},{oy:.4f}) -- ({px:.4f},{oy-.07:.4f});")
            lines.append(rf"\node[anchor=north] at ({px:.4f},{oy-.08:.4f}) {{{tick_label(x,panel.get('logx'),True)}}};")
        lines.append(rf"\draw[ipsGuide] ({ox:.4f},{oy+height:.4f}) -- ({ox:.4f},{oy:.4f}) -- ({ox+width:.4f},{oy:.4f});")
        lines.append(rf"\node at ({ox+width/2:.4f},{oy-.58:.4f}) {{{panel['xtex']}}};")
        lines.append(r"\begin{scope}")
        lines.append(rf"\clip ({ox:.4f},{oy:.4f}) rectangle ({ox+width:.4f},{oy+height:.4f});")
        for j, curve in enumerate(curves[i]):
            coordinates = " ".join(f"({xx:.4f},{yy:.4f})" for xx, yy in [xy(x, y) for x, y in curve])
            lines.append(rf"\draw[{TIKZ_COLORS[j]},line width=.95pt,{TIKZ_DASHES[j]}] plot coordinates {{{coordinates}}};")
        lines.append(r"\end{scope}")
    for j, tau in enumerate(temperatures):
        x = .65 + 3.22 * j
        lines.append(rf"\draw[{TIKZ_COLORS[j]},line width=1pt,{TIKZ_DASHES[j]}] ({x:.3f},-5.5) -- ({x+.5:.3f},-5.5);")
        lines.append(rf"\node[anchor=west] at ({x+.58:.3f},-5.5) {{$\tau={fmt(tau)}$}};")
    lines.append(rf"\node at (6.85,-5.82) {{$R={fmt(radius)}$; exact formulas; both axes in (d) are logarithmic.}};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_markup(panels, curves, temperatures, radius):
    width, height = 420, 215
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 730" role="img" '
             'aria-labelledby="title desc">',
             '<title id="title">Query and measure sensitivity of two-key softmax attention</title>',
             '<desc id="desc">Four exact plots compare temperatures. A sharper query transition and '
             'exponentially amplified rare mass occur at lower temperatures.</desc>',
             '<rect width="1080" height="730" fill="white"/>',
             f'<g font-family="system-ui,sans-serif" fill="{INK}" font-size="13">']
    for i, panel in enumerate(panels):
        ox, oy = 72 + (i % 2) * 535, 49 + (i // 2) * 320
        def xy(x, y):
            return (ox + width * (x - panel["xmin"]) / (panel["xmax"] - panel["xmin"]),
                    oy + height * (panel["ymax"] - y) / (panel["ymax"] - panel["ymin"]))
        parts.append(f'<text x="{ox+width/2}" y="{oy-19}" text-anchor="middle" font-size="16">{html.escape(panel["title"])}</text>')
        for y in panel["yticks"]:
            _, py = xy(panel["xmin"], y)
            parts.extend([f'<path d="M {ox} {py:.4f} H {ox+width}" stroke="{GUIDE}" stroke-opacity=".45"/>',
                          f'<text x="{ox-9}" y="{py+4:.4f}" text-anchor="end">{tick_label(y,panel.get("logy"))}</text>'])
        for x in panel["xticks"]:
            px, _ = xy(x, panel["ymin"])
            parts.append(f'<text x="{px:.4f}" y="{oy+height+23}" text-anchor="middle">{tick_label(x,panel.get("logx"))}</text>')
        parts.append(f'<path d="M {ox} {oy} V {oy+height} H {ox+width}" stroke="{GUIDE}" fill="none"/>')
        parts.append(f'<text x="{ox+width/2}" y="{oy+height+49}" text-anchor="middle">{html.escape(panel["xlabel"])}</text>')
        for j, curve in enumerate(curves[i]):
            d = " ".join(("M" if k == 0 else "L") + f" {px:.3f} {py:.3f}"
                         for k, (px, py) in enumerate(xy(x, y) for x, y in curve))
            parts.append(f'<path d="{d}" stroke="{COLORS[j]}" stroke-width="2.6" '
                         f'stroke-dasharray="{SVG_DASHES[j]}" fill="none"/>')
    for j, tau in enumerate(temperatures):
        x = 130 + j * 220
        parts.append(f'<path d="M {x} 696 h 38" stroke="{COLORS[j]}" stroke-width="2.8" stroke-dasharray="{SVG_DASHES[j]}"/>')
        parts.append(f'<text x="{x+46}" y="701">τ = {fmt(tau)}</text>')
    parts.append(f'<text x="540" y="727" text-anchor="middle">R = {fmt(radius)}; exact formulas. Panel (d) has logarithmic axes.</text></g></svg>')
    return "\n".join(parts)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two-key attention sensitivity</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;color:#263238;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#fafbfc}
h1{font-size:1.7rem}h2{font-size:1.15rem}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.1rem;background:white;padding:1rem;border:1px solid #d8dee5;border-radius:8px}
label{display:block}input{width:100%}output{font-variant-numeric:tabular-nums;font-weight:600}.readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:1rem 0}.readouts div{background:white;padding:.7rem;border:1px solid #d8dee5;border-radius:6px}.readouts span{display:block;color:#4e5962;font-size:.9rem}
svg{width:100%;height:auto;background:white}code{font-size:.9em}.note{color:#4e5962}button{padding:.3rem .7rem}a{color:#0072b2}
</style>
<h1>Query sensitivity and rare mass in softmax attention</h1>
<p>Two keys sit at −R and R, with R = <span id="radius"></span>. Panels (a)–(b) use equal masses.
Panels (c)–(d) keep the query at R and change the measure from δ<sub>−R</sub> to
(1−ε)δ<sub>−R</sub>+εδ<sub>R</sub>. The colored curves compare fixed temperatures;
the black dashed curve and dots track your selection.</p>
<div class="controls">
<label>Temperature τ: <output id="tauOut"></output><input id="tau" type="range" step="0.005" aria-label="logarithm of temperature"></label>
<label>Query x: <output id="xOut"></output><input id="x" type="range" step="0.005" aria-label="query"></label>
<label>Added mass ε: <output id="epsOut"></output><input id="eps" type="range" max="0" step="0.01" aria-label="base ten logarithm of added mass"></label>
</div>
<div class="readouts" aria-live="polite">
<div><span>Equal-mass field Φ(x)</span><output id="field"></output></div>
<div><span>Query derivative ∂Φ/∂x</span><output id="derivative"></output></div>
<div><span>Rare-mass field Φ<sub>ε</sub>(R)</span><output id="rareField"></output></div>
<div><span>Amplification |ΔΦ| / W₁</span><output id="gain"></output></div>
</div>
<div id="figure">__STATIC_SVG__</div>
<h2>What changes when temperature decreases?</h2>
<p>The largest query derivative is R²/τ, attained at the score tie x=0. At the fixed query R,
the added key receives half the attention once ε=1/(1+exp(2R²/τ)). Thus a sufficiently
small amount of new mass can change the field substantially. As ε tends to zero, its amplification
relative to W₁=2Rε tends to exp(2R²/τ). This is an exact two-key calculation, not a Monte Carlo estimate.</p>
<p class="note">All code and data are local. The plots use exact formulas sampled on a uniform query grid
and a logarithmic mass grid. CSV files retain the amplification in log₁₀ form to avoid overflow.
<a href="sensitivity_query.csv">Query CSV</a> · <a href="sensitivity_rare_mass.csv">Rare-mass CSV</a> ·
<a href="sensitivity.svg">Static SVG</a> · <a href="sensitivity_checks.json">Checks and settings</a></p>
<script>
"use strict";
const config=__CONFIG__, panels=__PANELS__;
const ns="http://www.w3.org/2000/svg", R=config.radius;
const $=id=>document.getElementById(id), nice=x=>Number(x).toPrecision(5);
const sig=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
function logadd(a,b){return Math.max(a,b)+Math.log1p(Math.exp(-Math.abs(a-b)));}
function query(x,t){const e=Math.exp(-2*Math.abs(R*x/t));return [R*Math.tanh(R*x/t),R*R/t*4*e/(1+e)**2];}
function rare(le,t){if(le>=0)return [R,0]; const lm=Math.log(-Math.expm1(le)),s=2*R*R/t;
return [R*(2*sig(le-lm+s)-1),-logadd(le,lm-s)/Math.LN10];}
function element(tag,attrs){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e;}
function coords(i,x,y){const p=panels[i];return [72+(i%2)*535+420*(x-p.xmin)/(p.xmax-p.xmin),49+Math.floor(i/2)*320+215*(p.ymax-y)/(p.ymax-p.ymin)];}
$("radius").textContent=nice(R);$("tau").min=Math.log(Math.min(...config.temperatures));$("tau").max=Math.log(Math.max(...config.temperatures));$("tau").value=Math.log(config.temperatures[0]);
$("x").min=-R;$("x").max=R;$("x").step=R/200;$("x").value=0;$("eps").min=config.log10_epsilon_min;$("eps").value=-4;
const svg=$("figure").querySelector("svg"), layer=element("g",{"aria-hidden":"true"});svg.appendChild(layer);
function update(){const tau=Math.exp(+$('tau').value),x=+$('x').value,le=+$('eps').value;
const q=query(x,tau),r=rare(le*Math.LN10,tau);$("tauOut").textContent=nice(tau);$("xOut").textContent=nice(x);$("epsOut").textContent=(10**le).toExponential(3);
$("field").textContent=nice(q[0]);$("derivative").textContent=nice(q[1]);$("rareField").textContent=nice(r[0]);$("gain").textContent=r[1]<300?nice(10**r[1]):"10^"+nice(r[1]);
layer.replaceChildren();for(let i=0;i<4;i++){let d="";for(let k=0;k<config.points;k++){const xx=panels[i].xmin+(panels[i].xmax-panels[i].xmin)*k/(config.points-1);
const yy=i<2?query(xx,tau)[i]:rare(xx*Math.LN10,tau)[i-2];const [px,py]=coords(i,xx,yy);d+=(k?" L ":"M ")+px+" "+py;}
layer.appendChild(element("path",{d,stroke:"#263238","stroke-width":2,"stroke-dasharray":"5 5",fill:"none"}));
const [cx,cy]=coords(i,i<2?x:le,i<2?q[i]:r[i-2]);layer.appendChild(element("circle",{cx,cy,r:5,fill:"#263238",stroke:"white","stroke-width":1.8}));}}
for(const id of["tau","x","eps"])$(id).addEventListener("input",update);update();
</script></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[.25, .5, 1, 2])
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--log10-epsilon-min", type=float, default=-8)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not math.isfinite(args.radius) or not 1e-4 <= args.radius <= 100:
        parser.error("radius must be finite and between 1e-4 and 100")
    if not 2 <= len(args.temperatures) <= 4 or any(not math.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("give two to four positive finite temperatures")
    if any(not 1e-8 <= args.radius**2 / t <= 1e4 for t in args.temperatures):
        parser.error("R^2/tau must be between 1e-8 and 1e4")
    if args.points < 51 or args.points > 2001:
        parser.error("points must be between 51 and 2001")
    if not math.isfinite(args.log10_epsilon_min) or not -300 <= args.log10_epsilon_min <= -1:
        parser.error("log10-epsilon-min must be between -300 and -1")
    temperatures = sorted(set(args.temperatures))
    if len(temperatures) < 2:
        parser.error("give at least two distinct temperatures")
    checks = verify(args.radius, temperatures)
    qrows, rrows, curves = build_data(args.radius, temperatures, args.points, args.log10_epsilon_min)
    panels = panel_specs(args.radius, temperatures, args.log10_epsilon_min)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, rows in [("query", qrows), ("rare_mass", rrows)]:
        with (args.output_dir / f"sensitivity_{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_tikz(args.output_dir / "sensitivity.tex", panels, curves, temperatures, args.radius)
    svg = svg_markup(panels, curves, temperatures, args.radius)
    (args.output_dir / "sensitivity.svg").write_text(svg + "\n", encoding="utf-8")
    config = {"radius": args.radius, "temperatures": temperatures, "points": args.points,
              "log10_epsilon_min": args.log10_epsilon_min, "method": "exact analytic formulas; no random sampling"}
    document = (HTML_TEMPLATE.replace("__STATIC_SVG__", svg).replace("__CONFIG__", json.dumps(config))
                .replace("__PANELS__", json.dumps(panels)))
    (args.output_dir / "sensitivity.html").write_text(document, encoding="utf-8")
    (args.output_dir / "sensitivity_checks.json").write_text(json.dumps({"configuration": config, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_directory": str(args.output_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

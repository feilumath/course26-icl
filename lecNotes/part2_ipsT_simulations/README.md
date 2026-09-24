# Part II: attention simulations and numerical examples

This folder contains the code and figures added in response to the three cyan
comments in `../part2_ipsT.tex`. The demonstrations use Python 3.10 or newer and
its standard library; no package installation or network connection is needed.

## Run and present

From the repository root:

```sh
python3 teaching/topic26Fall/part2_ipsT_simulations/run_all.py
```

Then open `index.html` in a browser. On macOS, from this folder:

```sh
open index.html
```

The sensitivity page has sliders for temperature, query, and added mass. The
dynamics page has Play/Pause, Reset, a time slider, a temperature selector, and
a field-direction toggle. Both pages embed their data and scripts and work
directly from disk, including during an offline lecture.

The scripts can also be run separately:

```sh
python3 sensitivity.py
python3 integrators.py
python3 dynamics.py
```

`run_all.py` runs all numerical checks and refreshes `figures/sensitivity.tex`
and `figures/dynamics.tex`, the native TikZ inputs included by the note. These
two figure inputs are retained alongside the source, so compiling the note
does not require running Python first. Other generated data and browser pages
are in `results/`, which follows the repository's existing ignore rules.

Compile the note from `teaching/topic26Fall/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error part2_ipsT.tex
```

## 1. Query sensitivity and rare mass

`sensitivity.py` evaluates the exact one-dimensional two-key example with
`R = 1` and temperatures `0.25, 0.5, 1, 2`. It plots:

- The equal-mass query field `R tanh(R x / tau)`.
- Its query derivative, whose maximum is `R²/tau` at `x = 0`.
- The field at query `R` after replacing `delta(-R)` by
  `(1-epsilon) delta(-R) + epsilon delta(R)`.
- The ratio of field change to `W1 = 2 R epsilon`, with limiting value
  `exp(2 R²/tau)` as `epsilon` tends to zero.

The query grid has 241 points. The rare-mass grid is uniform in `log10(epsilon)`
from `-8` to `0`. Logarithmic arithmetic avoids overflow and subtraction of
nearly equal field values. These are exact formulas sampled for plotting,
not Monte Carlo estimates.

Example with different settings, leaving the default lecture data intact:

```sh
python3 sensitivity.py --radius 0.8 --temperatures .2 .4 .8 1.6 --output-dir results/custom_sensitivity
```

The checks compare the formulas against weighted softmax, compare the analytic
query derivative against central differences, and verify the rare-mass limit,
amplification bounds, and half-attention threshold.

Outputs: `sensitivity.html`, `sensitivity.svg`, `sensitivity.tex`,
`sensitivity_query.csv`, `sensitivity_rare_mass.csv`, and
`sensitivity_checks.json`. The CSV stores amplification in base-ten logarithmic
form; the JSON records settings and check errors.

## 2. Time integrators

`integrators.py` implements the solved exercise
`dx_i/dt = mean(x) - x_i`. It advances the entire cloud with Euler, explicit
midpoint (RK2), classical RK4, implicit Euler, and implicit midpoint. The
implicit equations have an exact solution in this linear example; the script
does not claim that general implicit attention steps are this inexpensive.

Starting from `[-2, -0.5, 0.25, 1, 3]`, it compares the numerical solution at
`T = 2` with the exact solution, using 8, 16, 32, 64, and 128 steps. It verifies
mean preservation, the analytic centered-mode amplification factors, and the
expected orders 1, 2, 4, 1, and 2. A second panel uses `h = 2.5` to illustrate
how the size of the centered cloud changes over repeated steps. Decay at a
large step size does not establish an accurate approximation of `exp(-t)`.

Outputs: `integrators.svg`, `integrators_accuracy.csv`,
`integrators_stability.csv`, and `integrators_checks.json`.

## 3. Self-consistent identity-attention dynamics

`dynamics.py` solves the two deterministic finite-particle equations from the
note, with `Q = K = V = I`:

```text
normalized:   dx_i/dt = sum_j exp(dot(x_i,x_j)/tau) x_j
                        / sum_j exp(dot(x_i,x_j)/tau)

unnormalized: dx_i/dt = (1/N) sum_j exp(dot(x_i,x_j)/tau) x_j
```

Every Runge–Kutta stage rebuilds the empirical measure from all stage states.
The normalized drift uses shifted softmax. The unnormalized calculation
retains the actual exponential scale and the factor `1/N`. Neither model adds
noise, subtracts `x_i`, centers the cloud, projects onto a sphere, or clips the
drift. The field arrows in the animation show unit directions, not speeds.

The initial eight points are the four points below and their negatives:

```text
(0.80,  0.12), (0.65, -0.22), (0.45, 0.24), (0.30, -0.12)
```

The temperatures are `0.5, 1, 2`; the requested horizon is `T = 1.6`.
Adaptive RK4 uses step doubling, an absolute/relative scale
`2e-10 * (1 + max(abs(coordinates)))`, and maximum step `0.015`.
The local error estimate is the difference between one full step and two
half steps, divided by 15; the two-half-step state is accepted.

Each run stops at the first event `max_i |x_i| = 2.2`, or at the requested
horizon. Boundary crossings are localized by bisection. Oversized trial
stages are rejected and retried with a smaller step; accepted states are not
clipped. A cutoff does not assert finite-time blow-up or convergence. In
particular, normalized finite-particle attention satisfies
`max_i |x_i(t)| <= exp(t) max_i |x_i(0)|` and has no finite-time escape.

For the default cloud, the radius boundary is reached approximately at:

| Temperature | Normalized | Unnormalized |
| --- | --- | --- |
| 0.5 | 1.25272 | 0.32308 |
| 1 | Horizon reached | 0.99735 |
| 2 | Horizon reached | Horizon reached |

The two models use the same physical time in the animation. Once a run stops,
its panel explicitly identifies the last recorded time, displays that final
state, and hides field arrows. It does not draw a fictitious continuation.
The animation and plots use a separate `display_history`, retaining adaptive
steps and adding samples where the trajectory accelerates rapidly. Quarter,
midpoint, and three-quarter interpolation checks are below `5e-5` per particle.
Fixed-grid `history` is retained for the CSV and refinement comparisons.
The script also compares interpolation near each rapidly accelerating cutoff
against fresh tighter integrations, so a sparse display grid cannot hide an
otherwise accurate integration.

The recorded diagnostics are maximum and RMS radius, variance, mean, and
the interaction energy
`F = -tau/(2 N²) sum_ij exp(dot(x_i,x_j)/tau)`. In this symmetric example the
cloud spreads while the energy decreases. This is compatible with the theory:
the energy is unbounded below, and raw Euclidean identity attention has no
general clustering guarantee. Already a single particle solves `dx/dt = x`
in the normalized model and `dx/dt = exp(|x|²/tau) x` in the unnormalized one.

Checks include the finite-particle energy gradient and dissipation identity,
the exact normalized one-particle solution, and an independent scalar
quadrature for the unnormalized one-particle trajectory. Each default run is
repeated with tolerance `1e-11` and maximum step `0.0075`. The largest observed
particle discrepancy at shared output times is below `8e-8`; the boundary-time
discrepancy is below `2e-9`. These are refinement diagnostics, not a certified
error bound for arbitrary initial data or parameters.

Outputs: `dynamics.html`, `dynamics.svg`, `dynamics.tex`, `dynamics.csv`, and
`dynamics.json`. The JSON includes the initial cloud, settings, checks, stopping
status, fixed-grid trajectories, and display trajectories; the CSV includes per-particle states
and diagnostics. To change the experiment, edit the constants at the top of
`dynamics.py`, update any presentation limits that depend on them, and rerun the
checks. The supplied presentation is configured for the default experiment.

"""Radar bias experiment: the same pass and the same random noise, with and without a
constant elevation bias. Random noise averages away as readings pile up; a bias points
the same way every time, so the filter's error should drop and then flatten at a floor,
while its own uncertainty keeps shrinking as if nothing were wrong.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iss_now import load_iss
from iss_passes import find_passes, load_observer
from kalman_track import DT, SEED, run_ekf, simulate_pass

BIASES_DEG = [0.0, 0.1]
COLORS = ["#1f6feb", "#d1495b"]
PLOT_FILE = Path(__file__).parent / "output" / "bias_experiment.png"


def main():
    ts, iss = load_iss()
    cfg, site, tz = load_observer()
    p = max(find_passes(ts, iss, site, days=1), key=lambda p: p["peak"][1])
    print(f"ISS pass over {cfg['name']}: {p['rise'][0].astimezone(tz):%a %b %d %I:%M %p}, "
          f"peak {p['peak'][1]:.1f}°\n")
    print(f"{'bias':>6}  {'time':>6}  {'EKF error':>10}  {'EKF thinks':>10}  {'off by':>7}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for bias, color in zip(BIASES_DEG, COLORS):
        t, truth, sites, rots, meas, noise = simulate_pass(
            ts, iss, site, p, np.random.default_rng(SEED), bias_el_deg=bias)
        est, covs = run_ekf(sites, rots, meas, noise)
        err = np.linalg.norm(est[:, :3] - truth[:, :3], axis=1) * 1000
        sig = np.sqrt(np.trace(covs[:, :3, :3], axis1=1, axis2=2)) * 1000
        for s in [10, 60, 120, 240, 360]:
            if s < len(t):
                print(f"{bias:>5.1f}°  {s:>5}s  {err[s]:>8.0f} m  {sig[s]:>8.0f} m  "
                      f"{err[s] / sig[s]:>6.1f}×")
        secs = np.arange(len(t)) * DT
        ax.plot(secs, err, color=color, lw=2, label=f"{bias}° bias: actual error")
        ax.plot(secs, sig, "--", color=color, lw=1, label=f"{bias}° bias: what the filter thinks")
        print()

    ax.set_yscale("log"); ax.set_xlabel("seconds since rise"); ax.set_ylabel("position error (m)")
    ax.set_title(f"Constant elevation bias vs random noise — {cfg['name']}")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    PLOT_FILE.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT_FILE, dpi=130)
    print(f"Plot: {PLOT_FILE}")


if __name__ == "__main__":
    main()

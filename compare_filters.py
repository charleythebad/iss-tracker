"""EKF vs UKF on the same simulated radar data, across radars of decreasing quality.

Each scenario runs 30 Monte Carlo trials over the same ISS pass (same orbit, fresh
noise each trial) and reports accuracy, consistency (ANEES, ideal 3.0) and run time.
Uses the observer in config.local.json (or ISS_CONFIG) and the next day's highest pass.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iss_now import load_iss
from iss_passes import find_passes, load_observer
from kalman_track import run_ekf, simulate_pass
from ukf import run_ukf

RUNS = 30
SCENARIOS = [  # label, range noise (km), angle noise (deg), seconds between measurements
    ("Tracking radar, 1 s", 0.030, 0.03, 1.0),
    ("Cheap radar, 1 s", 0.5, 0.5, 1.0),
    ("Cheap radar, 10 s", 0.5, 0.5, 10.0),
    ("Very cheap radar, 5 s", 2.0, 2.0, 5.0),
]
FILTERS = {"EKF": run_ekf, "UKF": run_ukf}
PLOT_FILE = Path(__file__).parent / "output" / "ekf_vs_ukf.png"


def trial_errors(f, data, dt):
    t, truth, sites, rots, meas, noise = data
    start = time.perf_counter()
    est, covs = f(sites, rots, meas, noise, dt=dt)
    secs = time.perf_counter() - start
    e = est[2:, :3] - truth[2:, :3]
    nees = np.array([e[k] @ np.linalg.solve(covs[k + 2][:3, :3], e[k]) for k in range(len(e))])
    return np.linalg.norm(e, axis=1) * 1000, nees, secs


def run_scenario(ts, iss, site, p, sigma_range, sigma_angle, dt):
    out = {name: {"err": [], "nees": [], "secs": []} for name in FILTERS}
    for seed in range(RUNS):
        data = simulate_pass(ts, iss, site, p, np.random.default_rng(seed), dt=dt,
                             sigma_range_km=sigma_range, sigma_angle_deg=sigma_angle)
        for name, f in FILTERS.items():
            err, nees, secs = trial_errors(f, data, dt)
            out[name]["err"].append(err); out[name]["nees"].append(nees); out[name]["secs"].append(secs)
    return {name: {k: np.array(v) for k, v in r.items()} for name, r in out.items()}


def summarize(r):
    half = r["err"].shape[1] // 2
    rms = np.sqrt(np.mean(r["err"][:, half:] ** 2, axis=1))
    return np.median(rms), np.mean(r["nees"][:, half:]), np.mean(r["secs"]) * 1000


def main():
    ts, iss = load_iss()
    cfg, site, tz = load_observer()
    p = max(find_passes(ts, iss, site, days=1), key=lambda p: p["peak"][1])
    print(f"Pass over {cfg['name']}: {p['rise'][0].astimezone(tz):%a %b %d %I:%M %p}, "
          f"peak {p['peak'][1]:.1f}°, {RUNS} Monte Carlo runs per scenario\n")
    print("| Scenario | Filter | 2nd-half RMS error | ANEES (ideal 3.0) | Time per pass |")
    print("|---|---|---|---|---|")
    results = {}
    for label, sr, sa, dt in SCENARIOS:
        results[label] = run_scenario(ts, iss, site, p, sr, sa, dt)
        for name in FILTERS:
            rms, anees, ms = summarize(results[label][name])
            print(f"| {label} | {name} | {rms:,.0f} m | {anees:.2f} | {ms:.0f} ms |")

    label, _, _, dt = SCENARIOS[-1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"EKF": "#1f6feb", "UKF": "#d1242f"}
    for name, r in results[label].items():
        secs = (np.arange(r["err"].shape[1]) + 2) * dt
        ax1.plot(secs, np.median(r["err"], axis=0), color=colors[name], lw=2, label=f"{name} median error")
        ax2.plot(secs, np.mean(r["nees"], axis=0), color=colors[name], lw=1.5, label=name)
    ax1.set_yscale("log"); ax1.set_ylabel("position error (m)"); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title(f"EKF vs UKF — {label.lower()} (2 km / 2° noise), {RUNS} runs")
    ax2.axhline(3.0, color="black", ls="--", lw=1, label="ideal (3.0)")
    ax2.axhspan(2.19, 3.94, color="gray", alpha=0.15, label="95% band for 30 runs")  # chi2(90)/30
    ax2.set_yscale("log"); ax2.set_ylabel("ANEES"); ax2.set_xlabel("seconds since rise")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    PLOT_FILE.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT_FILE, dpi=130)
    print(f"\nPlot: {PLOT_FILE}")


if __name__ == "__main__":
    main()

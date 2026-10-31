"""Track the ISS through one pass with an Extended Kalman Filter fed by simulated radar.

Truth comes from SGP4 (the real ISS orbit). A simulated radar at the observer site
measures range, azimuth and elevation once per second with Gaussian noise. The EKF
knows none of the truth: it starts from the first two raw measurements, predicts
with two-body + J2 gravity, and corrects with each new measurement. Output is a
summary table and a plot of position error: raw measurements vs the filter.
"""
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iss_now import load_iss
from iss_passes import find_passes, load_observer

MU = 398600.4418          # Earth gravitational parameter, km^3/s^2
J2 = 1.08262668e-3
R_EARTH = 6378.137        # km
DT = 1.0                  # seconds between radar measurements
SIGMA_RANGE_KM = 0.030    # 30 m range noise
SIGMA_ANGLE_DEG = 0.03    # azimuth / elevation noise
SIGMA_ACCEL = 1e-6        # km/s^2 process noise: unmodeled drag and higher-order gravity
SEED = 7
PLOT_FILE = Path(__file__).parent / "output" / "kalman_error.png"


def gravity(r):
    """Two-body plus J2 acceleration in an Earth-centered inertial frame (km/s^2)."""
    x, y, z = r
    rn = np.linalg.norm(r)
    k = 1.5 * J2 * MU * R_EARTH**2 / rn**5
    zz = 5 * z * z / rn**2
    return -MU * r / rn**3 + k * np.array([x * (zz - 1), y * (zz - 1), z * (zz - 3)])


def propagate(x, dt):
    """RK4 step of the state [position km, velocity km/s]."""
    f = lambda s: np.concatenate([s[3:], gravity(s[:3])])
    k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def measure(r_sat, r_site, rot):
    """Range (km), azimuth and elevation (rad) of the satellite as seen from the site."""
    n, e, u = rot @ (r_sat - r_site)
    rng = np.sqrt(n * n + e * e + u * u)
    return np.array([rng, np.arctan2(e, n) % (2 * np.pi), np.arcsin(u / rng)])


def to_position(z, r_site, rot):
    """Invert one radar measurement back to an inertial position (no filtering)."""
    rng, az, el = z
    local = rng * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return r_site + rot.T @ local


def jacobian(fn, x, eps):
    fx = fn(x)
    cols = []
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps[i]
        cols.append((fn(x + dx) - fx) / eps[i])
    return np.array(cols).T


def residual(z, zhat):
    y = z - zhat
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi   # azimuth wraps at 360°
    return y


def simulate_pass(ts, iss, site, p, rng, dt=DT, sigma_range_km=SIGMA_RANGE_KM, sigma_angle_deg=SIGMA_ANGLE_DEG, bias_el_deg=0.0):
    """Truth states, site geometry and noisy measurements, once per dt, rise to set."""
    start = p["rise"][0].utc_datetime()
    secs = int((p["set"][0].utc_datetime() - start).total_seconds())
    t = ts.utc([start + timedelta(seconds=s) for s in np.arange(0, secs + 1, dt)])
    sat = iss.at(t)
    truth = np.vstack([sat.position.km, sat.velocity.km_per_s]).T
    sites = site.at(t).position.km.T
    rots = [site.rotation_at(t[i]) for i in range(len(t))]
    noise = np.array([sigma_range_km, np.radians(sigma_angle_deg), np.radians(sigma_angle_deg)])
    bias = np.array([0.0, 0.0, np.radians(bias_el_deg)])
    meas = [measure(truth[i, :3], sites[i], rots[i]) + bias + rng.normal(0, noise) for i in range(len(t))]
    return t, truth, sites, rots, np.array(meas), noise


def process_noise(dt):
    """White-acceleration process noise for a position/velocity state."""
    q = SIGMA_ACCEL**2
    return np.block([[q * dt**3 / 3 * np.eye(3), q * dt**2 / 2 * np.eye(3)],
                     [q * dt**2 / 2 * np.eye(3), q * dt * np.eye(3)]])


def initial_state(sites, rots, meas, noise, dt):
    """Start from the first two raw measurements only: position from the second fix,
    velocity from the difference. The starting covariance comes from the radar noise
    mapped into position space, so a noisier radar starts with honest uncertainty."""
    R = np.diag(noise**2)
    fixes, covs = [], []
    for i in (0, 1):
        fix = lambda z: to_position(z, sites[i], rots[i])
        J = jacobian(fix, meas[i], np.array([1e-6, 1e-9, 1e-9]))
        fixes.append(fix(meas[i])); covs.append(J @ R @ J.T)
    P = np.block([[covs[1], covs[1] / dt],
                  [covs[1] / dt, (covs[0] + covs[1]) / dt**2]])
    return np.concatenate([fixes[1], (fixes[1] - fixes[0]) / dt]), P


def run_ekf(sites, rots, meas, noise, dt=DT):
    R = np.diag(noise**2)
    Q = process_noise(dt)
    x, P = initial_state(sites, rots, meas, noise, dt)
    eps = np.array([1e-3] * 3 + [1e-6] * 3)
    est, covs = [np.full(6, np.nan), x.copy()], [np.full((6, 6), np.nan), P.copy()]
    for k in range(2, len(meas)):
        F = jacobian(lambda s: propagate(s, dt), x, eps)
        x = propagate(x, dt)
        P = F @ P @ F.T + Q
        h = lambda s: measure(s[:3], sites[k], rots[k])
        H = jacobian(h, x, eps)
        y = residual(meas[k], h(x))
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P
        est.append(x.copy()); covs.append(P.copy())
    return np.array(est), np.array(covs)


def main():
    ts, iss = load_iss()
    cfg, site, tz = load_observer()
    p = max(find_passes(ts, iss, site, days=1), key=lambda p: p["peak"][1])
    t, truth, sites, rots, meas, noise = simulate_pass(ts, iss, site, p, np.random.default_rng(SEED))
    est, covs = run_ekf(sites, rots, meas, noise)
    sig = np.sqrt(np.diagonal(covs[:, :3, :3], axis1=1, axis2=2))

    raw_err = np.array([np.linalg.norm(to_position(meas[i], sites[i], rots[i]) - truth[i, :3])
                        for i in range(len(t))]) * 1000
    ekf_err = np.linalg.norm(est[:, :3] - truth[:, :3], axis=1) * 1000
    vel_err = np.linalg.norm(est[:, 3:] - truth[:, 3:], axis=1) * 1000
    secs = np.arange(len(t)) * DT

    print(f"ISS pass over {cfg['name']}: {p['rise'][0].astimezone(tz):%a %b %d %I:%M %p}, "
          f"peak {p['peak'][1]:.1f}°, {len(t)} radar measurements @ {DT:.0f}s")
    print(f"Radar noise: range {SIGMA_RANGE_KM * 1000:.0f} m, angles {SIGMA_ANGLE_DEG}°\n")
    print(f"{'time':>6}  {'raw meas error':>15}  {'EKF pos error':>14}  {'EKF vel error':>14}")
    for s in [2, 5, 10, 30, 60, 120, 180, 240, 300, 360]:
        if s < len(t):
            print(f"{s:>5}s  {raw_err[s]:>13.0f} m  {ekf_err[s]:>12.0f} m  {vel_err[s]:>10.1f} m/s")
    tail = slice(len(t) // 2, None)
    print(f"\nSecond half of pass — RMS error: raw {np.sqrt(np.mean(raw_err[tail]**2)):.0f} m, "
          f"EKF {np.sqrt(np.nanmean(ekf_err[tail]**2)):.0f} m")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(secs, raw_err, ".", ms=2, color="#9aa5b1", label="raw radar fix (no filter)")
    ax.plot(secs, ekf_err, color="#1f6feb", lw=2, label="Extended Kalman Filter")
    ax.plot(secs, np.linalg.norm(sig, axis=1) * 1000, "--", color="#1f6feb", lw=1,
            label="EKF 1σ uncertainty (its own estimate)")
    ax.set_yscale("log"); ax.set_xlabel("seconds since rise"); ax.set_ylabel("position error (m)")
    ax.set_title(f"ISS tracking from {cfg['name']} — simulated radar, real orbit")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    PLOT_FILE.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT_FILE, dpi=130)
    print(f"Plot: {PLOT_FILE}")


if __name__ == "__main__":
    main()

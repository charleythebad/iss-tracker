"""UKF tests, plus a regression test for the initial-covariance bug the comparison found."""
import numpy as np
import pytest

from kalman_track import run_ekf, simulate_pass
from ukf import mean_measurement, run_ukf, sigma_points, sigma_weights

RUNS = 10
ANEES_LOW, ANEES_HIGH = 1.68, 4.70    # 95% bounds for the mean of 10 chi-square(3) draws


@pytest.fixture(scope="module")
def pass_(passes):
    return max(passes[:6], key=lambda p: p["peak"][1])


def late_anees(f, ts, iss, site, pass_, **noise):
    dt = noise.pop("dt", 1.0)
    nees = []
    for seed in range(RUNS):
        _, truth, sites, rots, meas, n = simulate_pass(ts, iss, site, pass_, np.random.default_rng(seed),
                                                       dt=dt, **noise)
        est, covs = f(sites, rots, meas, n, dt=dt)
        e = est[2:, :3] - truth[2:, :3]
        nees.append([e[k] @ np.linalg.solve(covs[k + 2][:3, :3], e[k]) for k in range(len(e))])
    anees = np.mean(nees, axis=0)
    return anees[len(anees) // 2:]


def test_sigma_points_reproduce_mean_and_covariance():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6)); P = A @ A.T + 6 * np.eye(6); x = rng.normal(size=6)
    lam, wm, wc = sigma_weights(6)
    X = sigma_points(x, P, lam)
    np.testing.assert_allclose(wm @ X, x, atol=1e-9)
    np.testing.assert_allclose((X - x).T @ (wc[:, None] * (X - x)), P, rtol=1e-6, atol=1e-6)


def test_azimuth_mean_is_circular():
    Z = np.array([[1000, np.radians(359), 0.5], [1000, np.radians(1), 0.5]])
    z = mean_measurement(Z, np.array([0.5, 0.5]))
    assert min(np.degrees(z[1]), 360 - np.degrees(z[1])) < 1e-6   # 0°, not 180°


def test_ukf_matches_ekf_on_a_good_radar(ts, iss, site, pass_):
    data = simulate_pass(ts, iss, site, pass_, np.random.default_rng(7))
    _, truth, sites, rots, meas, noise = data
    tail = slice(len(truth) // 2, None)
    rms = {}
    for name, f in (("ekf", run_ekf), ("ukf", run_ukf)):
        est, _ = f(sites, rots, meas, noise)
        rms[name] = np.sqrt(np.mean(np.sum((est[tail, :3] - truth[tail, :3]) ** 2, axis=1)))
    assert rms["ukf"] * 1000 < 100
    assert rms["ukf"] == pytest.approx(rms["ekf"], rel=0.1)


def test_ukf_is_statistically_consistent(ts, iss, site, pass_):
    anees = late_anees(run_ukf, ts, iss, site, pass_)
    assert np.mean((anees > ANEES_LOW) & (anees < ANEES_HIGH)) >= 0.90


def test_initial_covariance_scales_with_radar_noise(ts, iss, site, pass_):
    """Regression: a fixed 1 km / 1 km/s starting covariance left both filters ~100x
    overconfident on a 0.5° radar (ANEES ~290). It is now derived from the noise."""
    anees = late_anees(run_ekf, ts, iss, site, pass_, sigma_range_km=0.5, sigma_angle_deg=0.5)
    assert np.mean((anees > ANEES_LOW) & (anees < ANEES_HIGH)) >= 0.90


def test_ukf_stays_honest_where_ekf_linearization_breaks(ts, iss, site, pass_):
    noisy = dict(sigma_range_km=2.0, sigma_angle_deg=2.0, dt=5.0)
    ekf = late_anees(run_ekf, ts, iss, site, pass_, **dict(noisy))
    ukf = late_anees(run_ukf, ts, iss, site, pass_, **dict(noisy))
    assert ukf.mean() < ANEES_HIGH < ekf.mean()

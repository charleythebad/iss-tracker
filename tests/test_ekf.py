"""Filter tests: accuracy against raw fixes, and statistical consistency (NEES).

NEES (normalized estimation error squared) is e' P^-1 e for the position error e
and the filter's own position covariance P. For an honest filter it follows a
chi-square distribution with 3 degrees of freedom, so its average over many
Monte Carlo runs should sit near 3 (Bar-Shalom, Li & Kirubarajan, ch. 5.4).
"""
import numpy as np
import pytest

from kalman_track import run_ekf, simulate_pass, to_position

RUNS = 10
# 95% two-sided bounds for the mean of RUNS chi-square(3) draws: chi2(30)/10
ANEES_LOW, ANEES_HIGH = 1.68, 4.70


@pytest.fixture(scope="module")
def pass_(passes):
    return max(passes[:6], key=lambda p: p["peak"][1])


def run(ts, iss, site, pass_, seed):
    t, truth, sites, rots, meas, noise = simulate_pass(ts, iss, site, pass_, np.random.default_rng(seed))
    est, covs = run_ekf(sites, rots, meas, noise)
    return truth, sites, rots, meas, est, covs


def test_filter_beats_raw_fixes(ts, iss, site, pass_):
    truth, sites, rots, meas, est, _ = run(ts, iss, site, pass_, seed=7)
    tail = slice(len(truth) // 2, None)
    raw = np.array([to_position(meas[i], sites[i], rots[i]) for i in range(len(meas))])
    raw_rms = np.sqrt(np.mean(np.sum((raw[tail] - truth[tail, :3]) ** 2, axis=1)))
    ekf_rms = np.sqrt(np.mean(np.sum((est[tail, :3] - truth[tail, :3]) ** 2, axis=1)))
    assert ekf_rms < raw_rms / 5
    assert ekf_rms * 1000 < 100               # meters


def test_filter_is_statistically_consistent(ts, iss, site, pass_):
    nees = []
    for seed in range(RUNS):
        truth, _, _, _, est, covs = run(ts, iss, site, pass_, seed)
        e = est[2:, :3] - truth[2:, :3]
        nees.append([e[k] @ np.linalg.solve(covs[k + 2][:3, :3], e[k]) for k in range(len(e))])
    anees = np.mean(nees, axis=0)[len(nees[0]) // 2:]
    inside = np.mean((anees > ANEES_LOW) & (anees < ANEES_HIGH))
    assert inside >= 0.90, f"only {inside:.0%} of steps inside the 95% NEES bounds"

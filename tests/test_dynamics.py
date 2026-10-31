import numpy as np
import pytest

from kalman_track import J2, MU, R_EARTH, gravity, propagate


def state_at(iss, t):
    s = iss.at(t)
    return np.concatenate([s.position.km, s.velocity.km_per_s])


def energy(x):
    """Specific orbital energy including the J2 potential term (conserved by two-body + J2)."""
    r, v = x[:3], x[3:]
    rn = np.linalg.norm(r)
    j2_pot = MU * J2 * R_EARTH**2 / (2 * rn**3) * (3 * (r[2] / rn) ** 2 - 1)
    return v @ v / 2 - MU / rn + j2_pot


def test_gravity_at_iss_altitude():
    r = np.array([R_EARTH + 420.0, 0, 0])
    assert np.linalg.norm(gravity(r)) == pytest.approx(MU / r[0] ** 2, rel=2e-3)


def test_j2_is_a_small_correction():
    r = np.array([4000.0, 3000.0, 4000.0])
    two_body = -MU * r / np.linalg.norm(r) ** 3
    ratio = np.linalg.norm(gravity(r) - two_body) / np.linalg.norm(two_body)
    assert 1e-4 < ratio < 3e-3


def test_propagation_conserves_energy_and_polar_momentum(iss, t0):
    x = state_at(iss, t0)
    e0, hz0 = energy(x), np.cross(x[:3], x[3:])[2]
    for _ in range(5560):                      # about one orbit at 1 s steps
        x = propagate(x, 1.0)
    assert abs(energy(x) / e0 - 1) < 1e-9
    assert abs(np.cross(x[:3], x[3:])[2] / hz0 - 1) < 1e-9


def test_propagation_tracks_sgp4_over_two_minutes(ts, iss):
    t = ts.utc(2026, 10, 31, 0, 0, [0, 120])
    x = state_at(iss, t[0])
    for _ in range(120):
        x = propagate(x, 1.0)
    assert np.linalg.norm(x[:3] - iss.at(t[1]).position.km) * 1000 < 10   # meters

import numpy as np
import pytest

from iss_passes import compass
from kalman_track import measure, residual, to_position


def test_measure_matches_skyfield(iss, site, t0):
    rng, az, el = measure(iss.at(t0).position.km, site.at(t0).position.km, site.rotation_at(t0))
    alt_ref, az_ref, dist_ref = (iss - site).at(t0).altaz()
    assert rng == pytest.approx(dist_ref.km, abs=1e-6)
    assert np.degrees(az) == pytest.approx(az_ref.degrees, abs=1e-6)
    assert np.degrees(el) == pytest.approx(alt_ref.degrees, abs=1e-6)


def test_measurement_round_trip(site, t0):
    r_site, rot = site.at(t0).position.km, site.rotation_at(t0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = np.array([rng.uniform(400, 2500), rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 1.5)])
        np.testing.assert_allclose(measure(to_position(z, r_site, rot), r_site, rot), z, atol=1e-9)


def test_azimuth_residual_wraps_through_north():
    y = residual(np.array([1000.0, np.radians(1), 0.5]), np.array([1000.0, np.radians(359), 0.5]))
    assert np.degrees(y[1]) == pytest.approx(2.0)


@pytest.mark.parametrize("az, expected", [(0, "N"), (44, "NE"), (90, "E"), (200, "S"), (337.6, "N")])
def test_compass(az, expected):
    assert compass(az) == expected

import numpy as np

from iss_passes import DARK_SUN_ALT_DEG, MIN_ELEVATION_DEG, visible_window


def test_passes_are_well_formed(passes):
    assert 10 <= len(passes) <= 30            # ~5-6 per day above 10° at 39°N
    for p in passes:
        rise, peak, set_ = p["rise"][0].tt, p["peak"][0].tt, p["set"][0].tt
        assert rise < peak < set_
        assert p["peak"][1] >= MIN_ELEVATION_DEG
        assert 0.5 <= (set_ - rise) * 1440 <= 12  # minutes


def test_passes_are_one_orbit_apart(passes):
    gaps = np.diff([p["peak"][0].tt for p in passes]) * 1440
    assert gaps.min() > 80                    # never closer than ~one ISS orbit (92.9 min)


def test_visible_windows_are_sunlit_under_dark_sky(ts, iss, site, eph, passes):
    windows = [visible_window(ts, iss, site, eph, p) for p in passes]
    assert any(w is not None for w in windows), "expected some visible passes in 3 days"
    assert any(w is None for w in windows), "expected some passes in Earth's shadow or daylight"
    for w in filter(None, windows):
        mid = ts.tt_jd((w[0].tt + w[1].tt) / 2)
        sun_alt = (eph["earth"] + site).at(mid).observe(eph["sun"]).apparent().altaz()[0].degrees
        assert iss.at(mid).is_sunlit(eph)
        assert sun_alt < DARK_SUN_ALT_DEG


def test_daylight_pass_is_not_visible(ts, iss, site, eph, passes):
    daylight = [p for p in passes
                if (eph["earth"] + site).at(p["peak"][0]).observe(eph["sun"]).apparent().altaz()[0].degrees > 0]
    assert daylight, "fixture window should include a daytime pass"
    for p in daylight:
        assert visible_window(ts, iss, site, eph, p) is None

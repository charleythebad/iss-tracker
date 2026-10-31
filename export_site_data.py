"""Export the data behind docs/index.html (the interactive showcase page).

Uses the pinned TLE in tests/fixtures and the example observer, so the page is a
reproducible snapshot rather than whatever today's orbit happens to be.
Run: .venv/bin/python export_site_data.py   ->   docs/data.json
"""
import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from skyfield.api import EarthSatellite, Loader, load, wgs84

from compare_filters import RUNS, SCENARIOS, run_scenario, summarize
from iss_passes import MIN_ELEVATION_DEG, find_passes, visible_window
from kalman_track import measure, simulate_pass, to_position
from kalman_track import run_ekf
from ukf import run_ukf

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "data.json"
START = (2026, 10, 31)          # UTC start of the 3-day window
ISS_ALT_KM = 420.0


def r(x, n=2):
    """Round for JSON; NaN (no estimate yet) becomes null."""
    return [None if np.isnan(v) else v for v in np.round(np.asarray(x, dtype=float), n).tolist()]


def sun_alt(eph, site, t):
    return (eph["earth"] + site).at(t).observe(eph["sun"]).apparent().altaz()[0].degrees


def to_azel(positions, sites, rots):
    z = np.array([measure(p, s, R) for p, s, R in zip(positions, sites, rots)])
    return np.degrees(z[:, 1]), np.degrees(z[:, 2])


def main():
    ts = load.timescale()
    name, l1, l2 = (ROOT / "tests" / "fixtures" / "iss.tle").read_text().splitlines()[:3]
    iss = EarthSatellite(l1, l2, name.strip(), ts)
    cfg = json.loads((ROOT / "config.example.json").read_text())
    site = wgs84.latlon(cfg["lat"], cfg["lon"], elevation_m=cfg["elevation_m"])
    tz = ZoneInfo(cfg["timezone"])
    eph = Loader(str(ROOT / "data"))("de421.bsp")
    local = lambda t, fmt: t.astimezone(tz).strftime(fmt)

    passes = find_passes(ts, iss, site, days=3, t0=ts.utc(*START))
    pass_rows = []
    for p in passes:
        w = visible_window(ts, iss, site, eph, p)
        pass_rows.append({
            "day": local(p["rise"][0], "%a %b %-d"), "rise": local(p["rise"][0], "%-I:%M %p"),
            "peak_el": round(p["peak"][1], 1), "rise_az": round(p["rise"][2]), "set_az": round(p["set"][2]),
            "minutes": round((p["set"][0].tt - p["rise"][0].tt) * 1440, 1),
            "visible": None if w is None else [local(w[0], "%-I:%M"), local(w[1], "%-I:%M %p")],
            "peak_utc": p["peak"][0].utc_iso(),
        })
    visible = [p for p, row in zip(passes, pass_rows) if row["visible"]]
    show = max(visible[:3], key=lambda p: p["peak"][1])

    # Sky path of the showcase pass, every 5 s
    start = show["rise"][0].utc_datetime()
    secs = int((show["set"][0].utc_datetime() - start).total_seconds())
    t = ts.utc([start + timedelta(seconds=s) for s in range(0, secs + 1, 5)])
    alt, az, dist = (iss - site).at(t).altaz()
    sky = {"az": r(az.degrees), "el": r(alt.degrees), "range_km": r(dist.km, 0),
           "sunlit": iss.at(t).is_sunlit(eph).tolist(), "sun_alt": r(sun_alt(eph, site, t), 1),
           "time": [local(ti, "%-I:%M:%S %p") for ti in t]}

    # Ground track: one orbit before to one orbit after the showcase pass, every 20 s
    mid = show["peak"][0].utc_datetime()
    tg = ts.utc([mid + timedelta(seconds=s) for s in range(-5600, 5601, 20)])
    sub = wgs84.subpoint_of(iss.at(tg))
    galt = (iss - site).at(tg).altaz()[0].degrees
    ground = {"lat": r(sub.latitude.degrees), "lon": r(sub.longitude.degrees),
              "in_view": (galt >= MIN_ELEVATION_DEG).tolist(), "sunlit": iss.at(tg).is_sunlit(eph).tolist(),
              "time": [local(ti, "%-I:%M %p") for ti in tg]}
    e = np.radians(MIN_ELEVATION_DEG)
    footprint_deg = np.degrees(np.arccos(6378.137 / (6378.137 + ISS_ALT_KM) * np.cos(e)) - e)

    # Radar scenarios: one sample run for the sky view, 30-run statistics for the charts
    scenarios = []
    for label, sr, sa, dt in SCENARIOS:
        stats = run_scenario(ts, iss, site, show, sr, sa, dt)
        tt, truth, sites, rots, meas, noise = simulate_pass(ts, iss, site, show, np.random.default_rng(0),
                                                            dt=dt, sigma_range_km=sr, sigma_angle_deg=sa)
        raw_pos = np.array([to_position(m, s, R) for m, s, R in zip(meas, sites, rots)])
        sample = {"t": r(np.arange(len(tt)) * dt, 0)}
        for key, pos in (("truth", truth[:, :3]), ("raw", raw_pos)):
            a, el = to_azel(pos, sites, rots); sample[key] = {"az": r(a), "el": r(el)}
        for key, f in (("ekf", run_ekf), ("ukf", run_ukf)):
            est, _ = f(sites, rots, meas, noise, dt=dt)
            a, el = to_azel(est[2:, :3], sites[2:], rots[2:])
            sample[key] = {"az": r(np.r_[[np.nan, np.nan], a]), "el": r(np.r_[[np.nan, np.nan], el])}
        raw_err = np.linalg.norm(raw_pos - truth[:, :3], axis=1)[2:] * 1000   # the sample run's raw fixes
        summary = {k: dict(zip(("rms_m", "anees", "ms"), map(float, summarize(v)))) for k, v in stats.items()}
        scenarios.append({
            "label": label, "sigma_range_m": sr * 1000, "sigma_angle_deg": sa, "dt": dt,
            "t": r((np.arange(stats["EKF"]["err"].shape[1]) + 2) * dt, 0),
            "median_err_m": {k: r(np.median(v["err"], axis=0), 0) for k, v in stats.items()},
            "raw_err_m": r(raw_err, 0),
            "anees": {k: r(np.mean(v["nees"], axis=0), 2) for k, v in stats.items()},
            "summary": summary, "sample": sample,
        })
        print(f"{label}: EKF {summary['EKF']['rms_m']:.0f} m / {summary['EKF']['anees']:.2f}, "
              f"UKF {summary['UKF']['rms_m']:.0f} m / {summary['UKF']['anees']:.2f}")

    data = {
        "site": {"name": cfg["name"], "lat": cfg["lat"], "lon": cfg["lon"], "elevation_m": cfg["elevation_m"]},
        "tle_epoch": iss.epoch.utc_strftime("%Y-%m-%d %H:%M UTC"),
        "window": f"{local(ts.utc(*START), '%b %-d')}–{local(ts.utc(*START) + 3, '%b %-d, %Y')}",
        "runs": RUNS, "anees_band": [2.19, 3.94],
        "showcase": {"day": local(show["rise"][0], "%A %b %-d"), "rise": local(show["rise"][0], "%-I:%M %p"),
                     "peak_el": round(show["peak"][1], 1)},
        "passes": pass_rows, "sky": sky, "ground": ground, "footprint_deg": round(float(footprint_deg), 2),
        "scenarios": scenarios,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"{OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

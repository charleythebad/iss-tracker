"""Predict upcoming ISS passes over the observer in config.local.json.

A pass is visible when the ISS is sunlit while the observer's sky is dark
(sun below civil twilight). Use --visible to list only those.
"""
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from skyfield.api import Loader, wgs84

from iss_now import load_iss

# ISS_CONFIG overrides the observer file, e.g. ISS_CONFIG=config.example.json
CONFIG_FILE = Path(os.environ.get("ISS_CONFIG", Path(__file__).parent / "config.local.json"))
DAYS_AHEAD = 3
MIN_ELEVATION_DEG = 10.0
DARK_SUN_ALT_DEG = -6.0  # civil twilight
SAMPLE_SECONDS = 10


def load_observer():
    cfg = json.loads(CONFIG_FILE.read_text())
    site = wgs84.latlon(cfg["lat"], cfg["lon"], elevation_m=cfg["elevation_m"])
    return cfg, site, ZoneInfo(cfg["timezone"])


def compass(az_deg):
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return points[round(az_deg / 45) % 8]


def find_passes(ts, iss, site, days=DAYS_AHEAD, t0=None):
    t0 = ts.now() if t0 is None else t0
    t1 = ts.utc(t0.utc_datetime() + timedelta(days=days))
    times, events = iss.find_events(site, t0, t1, altitude_degrees=MIN_ELEVATION_DEG)
    passes, current = [], {}
    for t, event in zip(times, events):
        alt, az, _ = (iss - site).at(t).altaz()
        point = (t, alt.degrees, az.degrees)
        if event == 0:
            current = {"rise": point}
        elif event == 1 and current:
            current["peak"] = point
        elif event == 2 and "peak" in current:
            current["set"] = point
            passes.append(current)
            current = {}
    return passes


def visible_window(ts, iss, site, eph, p):
    """Return (start, end) of the part of a pass that is sunlit under a dark sky, or None."""
    start = p["rise"][0].utc_datetime()
    secs = int((p["set"][0].utc_datetime() - start).total_seconds())
    t = ts.utc([start + timedelta(seconds=s) for s in range(0, secs + 1, SAMPLE_SECONDS)])
    sunlit = iss.at(t).is_sunlit(eph)
    sun_alt = (eph["earth"] + site).at(t).observe(eph["sun"]).apparent().altaz()[0].degrees
    ok = sunlit & (sun_alt < DARK_SUN_ALT_DEG)
    if not ok.any():
        return None
    idx = ok.nonzero()[0]
    return t[idx[0]], t[idx[-1]]


def main():
    only_visible = "--visible" in sys.argv[1:]
    ts, iss = load_iss()
    cfg, site, tz = load_observer()
    eph = Loader(str(Path(__file__).parent / "data"))("de421.bsp")
    passes = find_passes(ts, iss, site)
    shown = 0
    print(f"ISS passes over {cfg['name']} — next {DAYS_AHEAD} days, above {MIN_ELEVATION_DEG:.0f}°"
          + (" (visible only)" if only_visible else ""))
    for p in passes:
        window = visible_window(ts, iss, site, eph, p)
        if only_visible and not window:
            continue
        shown += 1
        (tr, _, azr), (tp, altp, azp), (tset, _, azs) = p["rise"], p["peak"], p["set"]
        mins = (tset.utc_datetime() - tr.utc_datetime()).total_seconds() / 60
        seen = (f"VISIBLE {window[0].astimezone(tz):%I:%M}–{window[1].astimezone(tz):%I:%M %p}"
                if window else "not visible")
        print(f"  {tr.astimezone(tz):%a %b %d %I:%M %p}  rise {compass(azr):>2} → "
              f"peak {altp:4.1f}° {compass(azp):>2} at {tp.astimezone(tz):%I:%M %p} → "
              f"set {compass(azs):>2}  ({mins:.0f} min)  {seen}")
    if not shown:
        print("  none")


if __name__ == "__main__":
    main()

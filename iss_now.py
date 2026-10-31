"""Fetch the ISS orbital elements (TLE) from CelesTrak and print where it is right now."""
import sys
from pathlib import Path

from skyfield.api import load, wgs84

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle"
TLE_FILE = Path(__file__).parent / "data" / "iss.tle"
MAX_AGE_DAYS = 1.0


def load_iss():
    ts = load.timescale()
    fresh = not TLE_FILE.exists() or load.days_old(str(TLE_FILE)) >= MAX_AGE_DAYS
    sats = load.tle_file(TLE_URL, filename=str(TLE_FILE), reload=fresh)
    if not sats:
        sys.exit("No TLE returned from CelesTrak")
    return ts, sats[0]


def main():
    ts, iss = load_iss()
    now = ts.now()
    age_days = now - iss.epoch
    geo = iss.at(now)
    sub = wgs84.subpoint_of(geo)
    height = wgs84.height_of(geo)
    speed_kms = (sum(v * v for v in geo.velocity.km_per_s)) ** 0.5

    print(f"{iss.name}  (TLE epoch {iss.epoch.utc_strftime('%Y-%m-%d %H:%M UTC')}, {age_days:.2f} days old)")
    print(f"Time:      {now.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Position:  {sub.latitude.degrees:+.3f} lat, {sub.longitude.degrees:+.3f} lon")
    print(f"Altitude:  {height.km:.1f} km")
    print(f"Speed:     {speed_kms:.2f} km/s ({speed_kms * 2236.94:,.0f} mph)")


if __name__ == "__main__":
    main()

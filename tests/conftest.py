"""Shared fixtures. Everything is pinned: a saved TLE, a fixed start time and the
example observer, so results don't depend on today's orbit data or the network."""
import json
import sys
from pathlib import Path

import pytest
from skyfield.api import EarthSatellite, Loader, load, wgs84

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def ts():
    return load.timescale()


@pytest.fixture(scope="session")
def iss(ts):
    name, line1, line2 = (Path(__file__).parent / "fixtures" / "iss.tle").read_text().splitlines()[:3]
    return EarthSatellite(line1, line2, name.strip(), ts)


@pytest.fixture(scope="session")
def site():
    cfg = json.loads((ROOT / "config.example.json").read_text())
    return wgs84.latlon(cfg["lat"], cfg["lon"], elevation_m=cfg["elevation_m"])


@pytest.fixture(scope="session")
def t0(ts):
    return ts.utc(2026, 10, 31)


@pytest.fixture(scope="session")
def passes(ts, iss, site, t0):
    from iss_passes import find_passes
    return find_passes(ts, iss, site, days=3, t0=t0)


@pytest.fixture(scope="session")
def eph():
    path = ROOT / "data" / "de421.bsp"
    if not path.exists():
        pytest.skip("data/de421.bsp not downloaded yet (run iss_passes.py once)")
    return Loader(str(path.parent))("de421.bsp")

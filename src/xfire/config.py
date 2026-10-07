import json
from pathlib import Path

DEFAULT_REGIONS = {
    "Australia": (-44, -10, 112, 154),
    "Amazon": (-15, 5, -75, -45),
    "Indonesia": (-10, 6, 95, 141),
    "Central Africa": (-15, 10, 10, 40),
    "Bangladesh": (20.5, 26.7, 88.0, 92.8),
}
VIIRS_START = "2012-02-01"
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def load_regions(path=None):
    if not path:
        return DEFAULT_REGIONS.copy()
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in data.items()}

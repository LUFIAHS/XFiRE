import json
from pathlib import Path

DEFAULT_REGIONS = {}
# NOMINAL VIIRS (S-NPP) start, used ONLY to warn when the supplied VIIRS files start much later. The harmonization itself reads the
# real first VIIRS week from the processed data (results["viirs_start"]); nothing downstream hardcodes a transition date.
VIIRS_NOMINAL_START = "2012-02-01"
VIIRS_START = VIIRS_NOMINAL_START      # backwards-compatible alias
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def load_regions(path=None):
    if not path:
        return DEFAULT_REGIONS.copy()
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in data.items()}

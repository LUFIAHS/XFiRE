"""python tests/test_serve.py  - builds temporary SYNTHETIC inputs, starts the server, POSTs a drawn area, downloads hotspots."""
import json, tempfile, threading, urllib.request
from pathlib import Path
from test_helpers import write_sensor_test_csvs
from xfire.pipeline import run
from xfire.serve import make_server
with tempfile.TemporaryDirectory() as td:
    p = Path(td); m, v = write_sensor_test_csvs(p)
    run([m], [v], p/"out", use_xgb=False, log=lambda *a: None)
    httpd, eng = make_server(p/"out", 0); threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    post = lambda b: json.load(urllib.request.urlopen(urllib.request.Request(url + "/forecast", json.dumps(b).encode())))
    r = post({"aoi": [22, 24, 89, 91]})
    print("skill", {k: r["skill"].get(k) for k in ("model", "clim", "cover", "objective")})
    assert len(r["mean"]) == 26 and "lo80" in r and "deep" not in r
    assert "error" in post({"aoi": [0, 0.5, 0, 0.5]}) and "error" in post({"aoi": "x"})
    assert b"XFiRE" in urllib.request.urlopen(url).read()
    csv = urllib.request.urlopen(url + "/hotspots.csv?lat0=22&lat1=24&lon0=89&lon1=91&start=2015-06-01&end=2015-06-30").read().decode().splitlines()
    assert csv[0].startswith("latitude,longitude,acq_date,sensor") and len(csv) > 50 and all("2015-06" in r for r in csv[1:])
    print("serve test PASS")

"""Local app server (standard library only):
  GET  /               the dashboard
  GET  /hotspots.csv   raw hotspots inside an area and date range:  ?lat0=&lat1=&lon0=&lon1=&start=YYYY-MM-DD&end=YYYY-MM-DD[&limit=]
  POST /forecast       26-week outlook for ANY drawn area  {"aoi": [lat0, lat1, lon0, lon1], "method": "ratio" | "unet"}
"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from .forecast import forecast
from .harmonize import box_series, series_for, modis_last, week_weights_ratio
from . import hotspots as hs


class Engine:
    def __init__(self, res, out=None):
        self.r, self.n, self.first, self.lfw = res, res["nweeks"], res["first_w"], res["last_full_w"]
        self.lock, self.ml = threading.Lock(), modis_last(res)
        self.hot = Path(out) / "hotspots" if out and hs.available(Path(out) / "hotspots") else None
        self.weights = week_weights_ratio(res) if self.hot else None

    def forecast(self, box, method=None):
        with self.lock:
            M, V = box_series(self.r["cells"], self.n, box)
            if M.sum() + V.sum() < 2000:
                return {"error": "Too few detections in this area for a reliable forecast - draw a larger area."}
            h = series_for(self.r, box, method); f = forecast(h, self.first, self.lfw, params=self.r.get("forecast_params"), tune=False)
            if not f:
                return {"error": "Not enough history in this area for a forecast."}
            return f

    def csv_chunks(self, aoi, d0, d1, limit):
        """Raw hotspots (header first) for an area and date range, or None when the build kept no raw hotspots."""
        if self.hot is None: return None
        return hs.csv_chunks(hs.query(self.hot, aoi, d0, d1, limit), self.weights)


def _box(q):
    """[lat0, lat1, lon0, lon1] from the query string, or None for 'everywhere'."""
    try:
        v = [float(q[k][0]) for k in ("lat0", "lat1", "lon0", "lon1")]
    except (KeyError, ValueError):
        return None
    if not (-90 <= v[0] < v[1] <= 90 and -180 <= v[2] < v[3] <= 180): raise ValueError("area must satisfy -90<=lat0<lat1<=90, -180<=lon0<lon1<=180")
    return v


def make_server(out_dir, port=8000):
    out = Path(out_dir); res = json.loads((out / "results.json").read_text("utf-8"))
    eng = Engine(res, out); html = (out / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype, extra=None):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html", "/dashboard.html"): return self._send(200, html, "text/html; charset=utf-8")
            if u.path != "/hotspots.csv": return self._send(404, b"not found", "text/plain")
            try:
                q = parse_qs(u.query); aoi = _box(q); d0, d1 = (q.get(k, [None])[0] for k in ("start", "end"))
                limit = int(q.get("limit", [5_000_000])[0]); chunks = eng.csv_chunks(aoi, d0, d1, limit)
                if chunks is None: return self._send(404, b"this build kept no raw hotspots (built with --no-hotspots)", "text/plain")
                name = f"xfire_hotspots_{d0 or 'start'}_{d1 or 'end'}.csv"
                self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"'); self.end_headers()   # HTTP/1.0: body streams until close
                for piece in chunks: self.wfile.write(piece.encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try: self._send(400, f"bad request: {e}".encode(), "text/plain")
                except Exception: pass

        def do_POST(self):
            try:
                if self.path != "/forecast": return self._send(404, b"{}", "application/json")
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                aoi = [float(v) for v in body["aoi"]]
                if len(aoi) != 4 or aoi[0] >= aoi[1] or aoi[2] >= aoi[3]: raise ValueError("aoi must be [lat_min, lat_max, lon_min, lon_max]")
                m = body.get("method"); out_ = eng.forecast(aoi, m if m in ("ratio", "unet") else None)
            except Exception as e:                      # bad input must not kill the server
                out_ = {"error": f"bad request: {e}"}
            self._send(200, json.dumps(out_).encode(), "application/json")

        def log_message(self, *a): pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler), eng

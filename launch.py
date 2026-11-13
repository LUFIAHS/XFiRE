#!/usr/bin/env python3
"""Build and serve a global MODIS + VIIRS burning-activity calendar from NASA FIRMS CSV archives."""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def free_port(start):
    for port in range(start, start + 20):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-venv", action="store_true", help="use the currently active Python environment")
    parser.add_argument("--no-install", action="store_true", help="do not install missing core dependencies")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    parser.add_argument("--rebuild", action="store_true", help="force a new build from the FIRMS CSV files")
    parser.add_argument("--port", type=int, default=8000, help="local web-server port")
    args, extra = parser.parse_known_args()

    modis = sorted((ROOT / "data/modis").glob("*.csv"))
    viirs = sorted((ROOT / "data/viirs").glob("*.csv"))
    out = ROOT / "output"
    results = out / "results.json"
    has_real = bool(modis and viirs)
    if not has_real and not results.exists():
        print("\nXFiRE needs real NASA FIRMS CSV inputs before it can build the dashboard.")
        print("  1. Put MODIS active-fire CSV files in: data/modis/")
        print("  2. Put VIIRS active-fire CSV files in: data/viirs/")
        print("  3. Run run.bat again. The build retains global coverage for map-based area selection.")
        if not modis:
            print("\n  MODIS input: missing")
        if not viirs:
            print("  VIIRS input: missing")
        return 2

    python = sys.executable
    if not args.no_venv:
        venv_dir = ROOT / ".venv"
        if not venv_dir.exists():
            print("[1/4] creating virtual environment ...")
            venv.create(venv_dir, with_pip=True)
        python = str(venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))

    def has_modules(modules):
        return subprocess.call(
            [python, "-c", "import " + ",".join(modules)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0

    if not args.no_install:
        if not has_modules(["numpy", "pandas", "lightgbm", "xgboost", "hyperopt", "PIL", "sklearn"]):
            print("[2/4] installing core packages ...")
            if subprocess.call([python, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]) != 0:
                print("      Package install failed; trying to continue with what is installed.")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    stale = has_real and results.exists() and max(path.stat().st_mtime for path in modis + viirs) > results.stat().st_mtime
    build = args.rebuild or stale or not results.exists() or bool(extra)
    port = free_port(args.port)
    command = [python, "-m", "xfire", "--out", str(out), "--serve", "--port", str(port)]
    if build:
        if not has_real:
            print("Cannot rebuild: add real MODIS CSVs to data/modis and VIIRS CSVs to data/viirs first.")
            return 2
        command += ["--modis", *map(str, modis), "--viirs", *map(str, viirs), "--global"]
        print("[3/4] building from real FIRMS observations (global coverage) ...")
    else:
        print("[3/4] using the existing real-data build ...")

    print("[4/4] starting XFiRE ...")
    url = f"http://127.0.0.1:{port}"
    dash = out / "dashboard.html"

    def opener():
        # wait until the server really answers (a build can take many minutes), then open the browser
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        print(f"\nXFiRE is running: {url}\nStatic copy (no server needed): {dash}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    threading.Thread(target=opener, daemon=True).start()
    try:
        code = subprocess.call(command + extra, env=env, cwd=ROOT)
    except KeyboardInterrupt:
        return 0
    if code != 0:
        print(f"\nXFiRE stopped with an error (exit code {code}). See the messages above.")
        if dash.exists():
            print(f"A dashboard file was still produced; opening it directly: {dash}")
            if not args.no_browser:
                webbrowser.open(dash.as_uri())
    return code


if __name__ == "__main__":
    raise SystemExit(main())

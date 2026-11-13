import json
from pathlib import Path


def write_dashboard(results, path):
    here = Path(__file__).parent
    data = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    html = (here / "template.html").read_text("utf-8").replace("__DATA__", data)
    html = html.replace("__CORE__", (here / "core.js").read_text("utf-8"))
    Path(path).write_text(html, encoding="utf-8")

"""Run the full IIT Ropar data collection pipeline: python run_all.py [--skip api,render,crawl]"""
import subprocess
import sys
from pathlib import Path

STEPS = [("api", "fetch_api.py"), ("render", "render_site.py"), ("crawl", "crawl_subsites.py"),
         ("build", "build_corpus.py")]

skip = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
scraper = Path(__file__).parent / "scraper"
for name, script in STEPS:
    if name in skip:
        continue
    print(f"\n===== {name}: {script} =====", flush=True)
    subprocess.run([sys.executable, "-u", script], cwd=scraper, check=True,
                   env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})

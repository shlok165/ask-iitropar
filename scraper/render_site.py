"""Step 2: render every public page of the www.iitrpr.ac.in single-page app."""
import asyncio
import json
import re
from urllib.parse import urlparse

from common import BASE, RAW, get_json, make_doc, normalize_url, save_docs
from render import render_crawl

# Routes of the React app (taken from its JS bundle); admin/login routes and
# /on-roll-students (a list of individual students) are intentionally left out.
STATIC_ROUTES = """
/ /SoPs /about-institute /academic-calendar /academic-circulars /academic-fa /academic-people /acpgr /acug
/administrative-offices /admission-notifications /annual-reports /anti-ragging /board-governors /boards /bwc
/calendar /centres /ceoa /coe /convocation /convocation-archive /counselling-cell /curriculum /deans /departments
/director /equal-opportunity-cell /events /faculty-advisors /faculty-archive /faculty-positions /fee-structure
/fellowships /finance-committee /former-directors /guest-house /handbook-information /helpdesk /hindi-cell
/holidays /hostels /house-allotment-committee /institute-bus-facility /internal-complaints-committee
/internship-admissions /it-committee /jee-cell /jobs /newsletters /non-teaching-archive
/nonteaching-previous-year-question-papers /notifications-circular /ordinances /organisation-structure
/pg-admissions/pg-diploma /phd /phd-archive /pmrf /post-doctoral /post-doctoral-archive /project-positions
/project-positions-archive /rankings /research-awards /research-publications /rti /scholarships /senate /staff
/staff-positions /statutes-and-act /student-affairs /student-council /tenders /tenders-archive /vision-mission
/ug-admissions/btech /ug-admissions/bsc-bed /pg-admissions/MTech /pg-admissions/MSc /pg-admissions/dataScience
/aarohan /bost /it /kshitij
""".split()
PARAM_ROUTES = [r"/boards/[^/]+", r"/department/[^/]+(/contact|/faculty|/staff)?", r"/event/\d+", r"/news/\d+",
                r"/ug-admissions/[^/]+", r"/pg-admissions/[^/]+"]
SPA_RE = re.compile(r"^(" + "|".join([re.escape(r) for r in STATIC_ROUTES] + PARAM_ROUTES) + r")/?$")
MAIN_HOSTS = {"www.iitrpr.ac.in", "iitrpr.ac.in"}


def is_spa(url: str) -> bool:
    p = urlparse(url)
    return p.netloc in MAIN_HOSTS and bool(SPA_RE.match(p.path or "/")) and not p.query


def main():
    codes = [d["department_code"] for d in (get_json("get-departments") or {}).get("departments", [])]
    seeds = [BASE + r for r in STATIC_ROUTES]
    seeds += [f"{BASE}/department/{c}" for c in codes] + [f"{BASE}/department/{c}/contact" for c in codes]

    discovered: dict[str, list[str]] = {}

    def on_link(page_url, link):
        if not link.startswith("http") or is_spa(link):
            return
        discovered.setdefault(normalize_url(link), [])
        if len(discovered[normalize_url(link)]) < 5:
            discovered[normalize_url(link)].append(page_url)

    pages = asyncio.run(render_crawl(seeds, lambda u: is_spa(u) and urlparse(u).netloc == "www.iitrpr.ac.in"
                                     and not re.search(r"/(faculty|staff)$", u),  # covered by the API
                                     max_pages=1500, concurrency=5, on_link=on_link))

    docs = []
    for pg in pages:
        if len(pg["text"]) < 80:
            continue
        path = urlparse(pg["url"]).path.strip("/") or "home"
        title = pg["title"] if pg["title"] and pg["title"] != "IIT Ropar" else path.replace("-", " ").title()
        docs.append(make_doc(pg["url"], f"IIT Ropar — {title}", pg["text"], "site:www.iitrpr.ac.in"))
    save_docs("main_site", docs)

    (RAW / "discovered_links.json").write_text(json.dumps(discovered, indent=1), encoding="utf-8")
    print(f"[saved] {len(discovered)} outbound/static links -> data/raw/discovered_links.json")


if __name__ == "__main__":
    main()

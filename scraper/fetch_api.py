"""Step 1: pull the public JSON API behind www.iitrpr.ac.in and turn it into documents + CSVs."""
import csv
import html
import re
from collections import defaultdict
from datetime import date

from common import BASE, DATA, get_json, is_excluded, make_doc, save_docs, save_raw_json

SKIP_KEYS = {
    "id", "news_id", "created_at", "updated_at", "image", "images", "image_path", "status",
    "display_order", "publish", "show_on_main", "show_on_homepage", "banners", "mimeType",
    "mimetype", "external_key", "department_code", "dept_code",
}


def fmt(value, indent=0) -> list[str]:
    """Render nested JSON as readable 'Key: value' lines."""
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k in SKIP_KEYS or v in (None, "", [], {}):
                continue
            label = k.replace("_", " ").capitalize()
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{label}:")
                lines += fmt(v, indent + 1)
            else:
                lines.append(f"{pad}{label}: {strip_html(str(v))}")
    elif isinstance(value, list):
        for item in value:
            sub = fmt(item, indent + 1)
            if sub:
                lines.append(f"{pad}- " + sub[0].strip())
                lines += sub[1:]
    elif value not in (None, ""):
        lines.append(f"{pad}{strip_html(str(value))}")
    return lines


def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def file_links(files) -> list[str]:
    out = []
    for f in files or []:
        url = f.get("download_url") or ""
        if url and not url.startswith("http"):
            url = f"{BASE}/api/{url.lstrip('/').removeprefix('api/')}"
        if url and not is_excluded(url):
            out.append(f"- {f.get('name', 'file')}: {url}")
    return out


def write_csv(name, rows, fields):
    path = DATA / f"{name}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (strip_html(str(r.get(k) or ""))) for k in fields})
    print(f"[saved] {len(rows)} rows -> data/{name}.csv")


def main():
    docs = []

    depts = (get_json("get-departments") or {}).get("departments", [])
    faculty = (get_json("get-all-faculty") or {}).get("faculty", [])
    staff = (get_json("get-all-staff") or {}).get("staff", [])
    committees = (get_json("get-committees?page=1&limit=100") or {}).get("committees", [])
    profiles = (get_json("get-committee-profiles") or {}).get("profiles", [])
    for name, obj in [("departments", depts), ("faculty", faculty), ("staff", staff),
                      ("committees", committees), ("committee_profiles", profiles)]:
        save_raw_json(name, obj)

    dept_name = {d["department_code"]: d["department_name"] for d in depts}
    for p in faculty + staff:
        p["name"] = (p.get("name") or "").strip()
        p["department_name"] = dept_name.get(p.get("department"), p.get("department"))

    # --- Faculty: one doc per person ---
    for p in faculty:
        url = f"{BASE}/department/{p.get('department')}/faculty"
        body = [f"{p['name']} is a {p.get('designation') or 'faculty member'} in the "
                f"{p['department_name']} department at IIT Ropar.", ""]
        body += fmt({k: v for k, v in p.items() if k not in ("name", "role", "department")})
        docs.append(make_doc(url, f"Faculty: {p['name']} ({p['department_name']})", "\n".join(body),
                             "api:faculty", "faculty", person=p["name"], department=p["department_name"]))

    # --- Staff: one doc per person ---
    for p in staff:
        url = f"{BASE}/department/{p.get('department')}/staff"
        body = [f"{p['name']} works as {p.get('designation') or 'staff'} "
                f"({p['department_name']}) at IIT Ropar.", ""]
        body += fmt({k: v for k, v in p.items() if k not in ("name", "role", "department")})
        docs.append(make_doc(url, f"Staff: {p['name']} ({p['department_name']})", "\n".join(body),
                             "api:staff", "administration", person=p["name"], department=p["department_name"]))

    # --- Departments: overview + roster of faculty/staff ---
    fac_by, staff_by = defaultdict(list), defaultdict(list)
    for p in faculty:
        fac_by[p.get("department")].append(p)
    for p in staff:
        staff_by[p.get("department")].append(p)
    for d in depts:
        code = d["department_code"]
        body = [f"Department/Section: {d['department_name']} (code: {code})"]
        if d.get("dept_external_link"):
            body.append(f"Website: {d['dept_external_link']}")
        if d.get("about"):
            body += ["", strip_html(d["about"])]
        if fac_by[code]:
            body += ["", f"Faculty members ({len(fac_by[code])}):"]
            body += [f"- {p['name']}, {p.get('designation') or ''}"
                     + (f" — {p['specialization']}" if p.get("specialization") else "") for p in fac_by[code]]
        if staff_by[code]:
            body += ["", f"Staff ({len(staff_by[code])}):"]
            body += [f"- {p['name']}, {p.get('designation') or ''}" for p in staff_by[code]]
        docs.append(make_doc(f"{BASE}/department/{code}", f"Department: {d['department_name']}",
                             "\n".join(body), "api:departments", "departments"))

    # --- Committees with members ---
    members = defaultdict(list)
    for p in profiles:
        members[p.get("committee_code")].append(p)
    for c in committees:
        code = c.get("committee_code")
        body = [f"{c.get('committee_name')}: {strip_html(c.get('description') or '')}", "", "Members:"]
        body += [f"- {m.get('name')} — {m.get('designation') or m.get('role') or ''}"
                 + (f" ({m['department_name']})" if m.get("department_name") not in (None, "Not Applicable") else "")
                 for m in sorted(members[code], key=lambda m: m.get("display_order") or 0)]
        docs.append(make_doc(f"{BASE}/{code}", f"Committee: {c.get('committee_name')}", "\n".join(body),
                             "api:committees", "administration"))

    # --- News / events / announcements (global + per department) ---
    news = {}
    for path in ["get-news", "get-public-global-news", "get-limited-news"] + [f"get-dept-news/{c}" for c in dept_name]:
        for n in (get_json(path) or {}).get("news", []) or []:
            news[n.get("news_id")] = n
    save_raw_json("news", list(news.values()))
    for n in news.values():
        body = fmt({k: v for k, v in n.items() if k not in ("event_files",)})
        body += file_links(n.get("event_files"))
        url = f"{BASE}/event/{n['news_id']}" if n.get("type") == "event" else f"{BASE}/news/{n['news_id']}"
        docs.append(make_doc(url, f"{(n.get('type') or 'news').title()}: {n.get('title')}", "\n".join(body),
                             "api:news", "news-events", date=n.get("event_date") or n.get("created_at")))

    # --- Institute calendar (holidays, academic dates) ---
    today = date.today()
    events = {}
    for y in range(today.year - 3, today.year + 1):  # the API only accepts ~1-year windows
        for e in (get_json(f"calendar-events?from={y}-07-01&to={y + 1}-06-30") or {}).get("events", []):
            events[e["id"]] = e
    events = list(events.values())
    save_raw_json("calendar_events", events)
    by_year = defaultdict(list)
    for e in events:
        by_year[e.get("academic_year") or "unknown"].append(e)
    for ay, evs in by_year.items():
        evs.sort(key=lambda e: e.get("start_date") or "")
        lines = [f"- {str(e.get('start_date'))[:10]}"
                 + (f" to {str(e['end_date'])[:10]}" if e.get("end_date") else "")
                 + f": {e.get('title')} [{e.get('category')}]"
                 + (f" — {e['description']}" if e.get("description") else "")
                 + (f" @ {e['venue']}" if e.get("venue") else "") for e in evs]
        docs.append(make_doc(f"{BASE}/calendar", f"IIT Ropar Institute Calendar {ay}",
                             f"Institute calendar events for academic year {ay}:\n" + "\n".join(lines),
                             "api:calendar", "academics"))
    write_csv("calendar", events, ["start_date", "end_date", "title", "category", "description", "venue", "academic_year"])

    # --- Admissions, jobs/advertisements, tenders ---
    for path, key, cat, label, link in [
        ("admissions/active?status=1", "admissions", "admissions", "Admission", "/admission-notifications"),
        ("v2/public/advertisements?status=1", None, "administration", "Job/Position", "/jobs"),
        ("get-tenders?status=1", "tenders", "administration", "Tender", "/tenders"),
    ]:
        data = get_json(path) or {}
        items = data.get(key) if key else next((v for v in data.values() if isinstance(v, list)), [])
        save_raw_json(path.split("?")[0].replace("/", "_"), items or [])
        for it in items or []:
            body = fmt({k: v for k, v in it.items() if k != "files"}) + file_links(it.get("files"))
            title = it.get("title") or it.get("advertisement_title") or it.get("name") or label
            docs.append(make_doc(f"{BASE}{link}#{it.get('id')}", f"{label}: {title}", "\n".join(body), f"api:{cat}", cat))

    save_docs("api", docs)
    write_csv("faculty", faculty, ["name", "designation", "department_name", "specialization", "email",
                                   "phone", "office_address", "profile_link"])
    write_csv("staff", staff, ["name", "designation", "department_name", "email", "phone", "office_address"])
    write_csv("departments", depts, ["department_name", "department_code", "dept_external_link", "about"])
    write_csv("committee_members", profiles, ["committee_name", "name", "designation", "role", "department_name", "email"])


if __name__ == "__main__":
    main()

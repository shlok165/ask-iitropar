"""Exact lookups over the structured CSVs: faculty, staff, committee members, departments."""
import csv
import re
from datetime import datetime

from rapidfuzz import fuzz

from app import config
from app.search import tokenize
from app.sources import Source

BASE = "https://www.iitrpr.ac.in"
MAX_ROWS = 25        # cap on rows sent to the LLM
GROUP_ABOVE = 8      # more rows than this are sent as one combined source

# Extra names students use for departments/offices, keyed by department_code in departments.csv.
DEPT_ALIASES = {
    "cse": ["cse", "cs", "computer science", "computer"],
    "ee": ["ee", "electrical"],
    "mechanical": ["me", "mech", "mechanical"],
    "civil": ["civil", "ce"],
    "chemical": ["chemical", "che", "chemical engg"],
    "chemistry": ["chemistry", "chem", "cy"],
    "physics": ["physics", "ph"],
    "maths": ["maths", "math", "mathematics", "mnc", "ma"],
    "hss": ["hss", "humanities", "social sciences", "english", "economics", "psychology"],
    "saide": ["saide", "aide", "ai", "data engineering", "artificial intelligence"],
    "dbme": ["dbme", "bme", "cbme", "biomedical"],
    "mme": ["mme", "metallurgy", "metallurgical", "materials"],
    "hostel-management": ["hostel office", "hostel management", "hostel section", "caretaker"],
    "library": ["library", "librarian"],
    "medical-centre": ["medical centre", "medical center", "medical", "hospital", "doctor", "doctors"],
    "caps": ["caps", "placement", "placements", "placement cell", "cdpc", "training and placement"],
    "sports": ["sports office", "sports"],
    "academics": ["academic section", "academics office", "academics"],
    "accounts": ["accounts", "fee", "fees"],
    "it-cell": ["it section", "it cell", "it office", "internet", "wifi", "email account"],
    "registrar-office": ["registrar"],
    "director-office": ["director"],
    "sa-hm": ["student affairs"],
    "rnd": ["r&d", "research and development"],
    "work-estate": ["estate", "maintenance"],
    "jee-cell": ["jee cell", "jee"],
    "international-relations": ["international relations", "international"],
    "snehita-well-being": ["snehita", "counselling", "counseling", "wellbeing", "well being"],
    "facility": ["facilities section", "transport", "bus"],
}
ACADEMIC_DEPTS = {"cse", "ee", "mechanical", "civil", "chemical", "chemistry", "physics", "maths", "hss",
                  "saide", "dbme", "mme"}

# committee_name -> page on the main site, as used by the scraper
COMMITTEE_URLS = {
    "ACPGR": "acpgr", "ACUG": "acug", "Board of Academic Affairs": "boaa", "Board of Cultural Activities": "boca",
    "Board of Hostel Affairs": "boha", "Board of Literacy Activities": "bola", "Board of Science & Technology": "bost",
    "Board of Wellness Affairs": "bowa", "Board of Sports Affairs": "bosa", "Electoral Literacy Club": "electoral-club",
    "Equal Opportunity Cell": "equal-opportunity-cell", "Fitness Club / ODAC": "fitness-club",
    "Hostel Management Committee": "hm-committee", "House Allotment Committee": "house-allotment-committee",
    "IT Committee": "it-committee", "Jammu & Kashmir Cell": "jk-cell", "Journalism Club": "journalism-club",
    "Snehita Wellbeing Cell": "snehita-well-being",
}
COMMITTEE_ALIASES = {
    "Board of Cultural Activities": ["cultural board", "cultural"],
    "Board of Science & Technology": ["technical board", "tech board", "science and technology"],
    "Board of Sports Affairs": ["sports board"],
    "Board of Hostel Affairs": ["hostel board", "hostel affairs"],
    "Board of Literacy Activities": ["literary board", "literary", "literacy"],
    "Board of Academic Affairs": ["academic board", "academic affairs"],
    "Board of Wellness Affairs": ["wellness board", "wellness"],
    "Hostel Management Committee": ["warden", "wardens", "chief warden", "hmc"],
    "Fitness Club / ODAC": ["odac", "fitness club"],
    "Snehita Wellbeing Cell": ["snehita"],
    "Jammu & Kashmir Cell": ["j&k cell", "jk cell"],
    "Equal Opportunity Cell": ["eoc", "liaison officer"],
}

# Words that say *what kind* of lookup it is, not *what topic* to match.
LOOKUP_WORDS = set(
    "who whos list all professor professors prof profs faculty faculties teacher teachers teaches teach teaching "
    "staff members member email emails mail phone number contact contacts office room details name names "
    "department dept iit ropar iitrpr give tell me show please find get working works work research researching "
    "area areas interest interests field fields specialization specialisation expertise currently present "
    "board club cell committee head".split())
ROLE_WORDS = ["faculty advisor", "associate faculty advisor", "general secretary", "gen sec", "gsec", "chairperson",
              "chairman", "member secretary", "chief warden", "associate warden", "warden", "president",
              "sports officer", "liaison officer", "caretaker"]


def _read(name: str) -> list[dict]:
    with (config.ROOT / "data" / f"{name}.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _board_acronym(name: str) -> str | None:
    """'Board of Cultural Activities' -> 'boca' (how students write it)."""
    m = re.match(r"board of (.+)", name.lower())
    return "bo" + "".join(w[0] for w in re.findall(r"[a-z]+", m.group(1))) if m else None


class TableStore:
    def __init__(self):
        self.faculty = _read("faculty")
        self.staff = _read("staff")
        self.committees = _read("committee_members")
        self.departments = _read("departments")
        self.fetched = datetime.fromtimestamp((config.ROOT / "data" / "faculty.csv").stat().st_mtime).strftime("%Y-%m-%d")
        self.dept_by_code = {d["department_code"]: d["department_name"] for d in self.departments}
        self.dept_code_by_name = {v: k for k, v in self.dept_by_code.items()}
        self.dept_aliases = {code: set(DEPT_ALIASES.get(code, [])) | {code, name.lower()}
                             for code, name in self.dept_by_code.items() if code != "na"}
        self.committee_aliases = {}
        for name in COMMITTEE_URLS:
            aliases = {name.lower(), *COMMITTEE_ALIASES.get(name, [])}
            if acr := _board_acronym(name):
                aliases.add(acr)
            self.committee_aliases[name] = aliases
        self.people = {r["name"] for r in self.faculty + self.staff + self.committees if r["name"]}

    # --- entity detection --------------------------------------------------------
    def find_departments(self, q: str) -> list[str]:
        ql = q.lower()
        hits = [(code, max(len(a) for a in aliases if _has(ql, a)))
                for code, aliases in self.dept_aliases.items() if any(_has(ql, a) for a in aliases)]
        return [code for code, _ in sorted(hits, key=lambda h: -h[1])]

    def find_committees(self, q: str) -> list[str]:
        ql = q.lower()
        return [name for name, aliases in self.committee_aliases.items() if any(_has(ql, a) for a in aliases)]

    def find_people(self, q: str) -> list[str]:
        """Names of faculty/staff/office-bearers mentioned in the question (fuzzy, surname-aware)."""
        ql = re.sub(r"\b(dr|prof|mr|ms|mrs|sh|smt)\.?\s", " ", q.lower())
        out = []
        for name in self.people:
            parts = [p for p in re.findall(r"[a-z]+", re.sub(r"\b(dr|prof|mr|ms|mrs|sh|smt)\b\.?", "", name.lower()))
                     if len(p) > 2]
            if len(parts) >= 2 and sum(_has(ql, p) for p in parts) >= 2:
                out.append(name)
            elif parts and fuzz.partial_ratio(" ".join(parts), ql) >= 92 and len(" ".join(parts)) > 8:
                out.append(name)
        return out

    def find_roles(self, q: str) -> list[str]:
        ql = q.lower()
        return [r for r in ROLE_WORDS if _has(ql, r)]

    # --- lookup ------------------------------------------------------------------
    def lookup(self, question: str) -> list[Source]:
        depts, committees = self.find_departments(question), self.find_committees(question)
        people, roles = self.find_people(question), self.find_roles(question)
        ql = question.lower()
        alias_words = {w for code in depts for a in self.dept_aliases[code] for w in tokenize(a)}
        alias_words |= {w for c in committees for a in self.committee_aliases[c] for w in tokenize(a)}
        topic = [t for t in tokenize(question) if t not in LOOKUP_WORDS and t not in alias_words
                 and not any(t in tokenize(r) for r in roles)]
        sources: list[Source] = []

        if people:
            sources += self._people_rows(people)
        if committees or roles:
            sources += self._committee_rows(committees, roles)
        if not people and not committees and not roles:
            staff_intent = any(d not in ACADEMIC_DEPTS for d in depts) or re.search(r"\b(staff|office|section)\b", ql)
            if staff_intent:
                sources += self._staff_rows(depts, topic)
            if not staff_intent or not sources:
                sources += self._faculty_rows([d for d in depts if d in ACADEMIC_DEPTS], topic)
        return sources

    def _faculty_rows(self, depts: list[str], topic: list[str]) -> list[Source]:
        rows = [r for r in self.faculty if not depts or self.dept_code_by_name.get(r["department_name"]) in depts]
        if topic:
            phrase = " ".join(topic)
            scored = []
            for r in rows:
                hay = f"{r['specialization']} {r['designation']}".lower()
                score = 3 if phrase in hay else sum(_has(hay, t) for t in topic)
                if score:
                    scored.append((score, r))
            # keep only the best matches (all words / phrase beat partial matches)
            best = max((s for s, _ in scored), default=0)
            rows = [r for s, r in sorted(scored, key=lambda x: -x[0]) if s >= min(best, len(topic))]
        elif not depts:
            return []  # "list all professors" with no department or topic is too broad
        return self._to_sources(rows, "faculty", depts, topic)

    def _staff_rows(self, depts: list[str], topic: list[str]) -> list[Source]:
        rows = [r for r in self.staff if self.dept_code_by_name.get(r["department_name"]) in depts]
        if topic:
            matched = [r for r in rows if any(_has(r["designation"].lower(), t) for t in topic)]
            rows = matched or rows
        return self._to_sources(rows, "staff", depts, topic)

    def _committee_rows(self, committees: list[str], roles: list[str]) -> list[Source]:
        rows = [r for r in self.committees if not committees or r["committee_name"] in committees]
        if roles:
            rows = [r for r in rows if any(_has(f"{r['role']} {r['designation']}".lower(), x) for x in roles)] or rows
        out = []
        for name in dict.fromkeys(r["committee_name"] for r in rows):
            members = [r for r in rows if r["committee_name"] == name][:MAX_ROWS]
            lines = [self._person_line(r, committee=True) for r in members]
            out.append(Source(f"Committee: {name}", f"{BASE}/{COMMITTEE_URLS.get(name, 'committees')}", "api",
                              self.fetched, f"{name} members:\n" + "\n".join(lines)))
        return out

    def _people_rows(self, names: list[str]) -> list[Source]:
        out = []
        for r in self.faculty:
            if r["name"] in names:
                out.append(Source(f"Faculty: {r['name']} ({r['department_name']})", self._faculty_url(r), "api",
                                  self.fetched, self._person_line(r)))
        for r in self.staff:
            if r["name"] in names:
                out.append(Source(f"Staff: {r['name']} ({r['department_name']})", self._staff_url(r), "api",
                                  self.fetched, self._person_line(r)))
        for r in self.committees:
            if r["name"] in names:
                out.append(Source(f"Committee: {r['committee_name']}",
                                  f"{BASE}/{COMMITTEE_URLS.get(r['committee_name'], 'committees')}", "api",
                                  self.fetched, self._person_line(r, committee=True)))
        return out

    # --- formatting --------------------------------------------------------------
    def _faculty_url(self, r: dict) -> str:
        code = self.dept_code_by_name.get(r["department_name"])
        return r["profile_link"] or (f"{BASE}/department/{code}/faculty" if code else f"{BASE}/faculty")

    def _staff_url(self, r: dict) -> str:
        code = self.dept_code_by_name.get(r["department_name"])
        return f"{BASE}/department/{code}/staff" if code else f"{BASE}/staff"

    @staticmethod
    def _person_line(r: dict, committee: bool = False) -> str:
        fields = [("role", "Role"), ("designation", "Designation"), ("department_name", "Department"),
                  ("specialization", "Research areas"), ("email", "Email"), ("phone", "Office phone"),
                  ("office_address", "Office")]
        bits = [f"{label}: {r[k]}" for k, label in fields
                if r.get(k) and r[k] != "Not Applicable" and not (committee and k == "designation" and r[k] == r.get("role"))]
        return f"- {r['name']}: " + "; ".join(bits)

    def _to_sources(self, rows: list[dict], kind: str, depts: list[str], topic: list[str]) -> list[Source]:
        rows = rows[:MAX_ROWS]
        url_of = self._faculty_url if kind == "faculty" else self._staff_url
        if len(rows) <= GROUP_ABOVE:
            return [Source(f"{kind.title()}: {r['name']} ({r['department_name']})", url_of(r), "api", self.fetched,
                           self._person_line(r)) for r in rows]
        dept_names = ", ".join(self.dept_by_code[d] for d in depts) or "all departments"
        about = f" matching '{' '.join(topic)}'" if topic else ""
        url = f"{BASE}/department/{depts[0]}/{kind}" if len(depts) == 1 else f"{BASE}/{kind}"
        return [Source(f"{kind.title()} of {dept_names}{about} ({len(rows)})", url, "api", self.fetched,
                       "\n".join(self._person_line(r) for r in rows))]

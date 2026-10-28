"""Rule-based router: decides which data path answers a question. No LLM call, so it costs nothing.

Order: privacy guard -> live API (time-sensitive) -> CSV tables (list/lookup) -> hybrid retrieval.
"""
import re
from dataclasses import dataclass, field

from app.tables import TableStore

REFUSAL = ("I can't look up or share personal details about individual students. "
           "For student matters, please contact the relevant office or board through its official email.")

# --- Privacy guard ------------------------------------------------------------------------
ENTRY_NO = re.compile(r"\b20\d{2}\s?[a-z]{3}\s?\d{4}\b", re.I)
# personal attributes nobody should be profiled by
ATTR = (r"(phone|phone number|mobile|mobile number|contact number|whatsapp|instagram|insta|snapchat|linkedin|address|"
        r"home address|room|room number|cgpa|sgpa|gpa|grades?|marks|roll number|roll no|entry number|entry no|"
        r"enrol+ment number|dob|date of birth|hometown|girlfriend|boyfriend)")
# a few attributes that are personal even when the question is about "students" in general
STRICT_ATTR = (r"(phone|phone number|mobile|mobile number|contact number|whatsapp|instagram|insta|snapchat|"
               r"home address|room number|roll number|roll no|entry number|entry no|enrol+ment number|dob|"
               r"date of birth|hometown)")
STUDENT = r"(student|students|batchmate|batchmates|classmate|classmates|roommate|junior|juniors|senior|seniors|fresher|freshers)"
# anything official the attribute might belong to instead of a person
OFFICIAL = re.compile(
    r"\b(iit|ropar|institute|campus|college|hostel|office|department|dept|section|cell|board|club|council|committee|"
    r"warden|caretaker|faculty|professor|prof|dr|staff|hod|director|registrar|dean|library|medical|security|"
    r"jee|gate|exam|admission|interview|placement|election|semester|course|mess|canteen|bank|post office|"
    r"general secretary|secretary|president|representative|coordinator|advisor)\b", re.I)
OFFICE_BEARER = re.compile(r"council|representative|secretary|gsec|gen sec|club|board|coordinator|office.?bearer|"
                           r"president|captain|head|statistics|stats|how many|number of|count", re.I)


# "tell me all you know about <name>", "who is <name>", "details of <name>"
PROFILE = re.compile(
    r"\b(?:tell\s+(?:me\s+)?(?:all\s+|everything\s+)?(?:you\s+know\s+)?about|(?:all|everything)\s+you\s+know\s+about|"
    r"who\s+is|who's|info(?:rmation)?\s+(?:on|about)|details\s+(?:of|about|on)|profile\s+of|background\s+of|"
    r"search\s+(?:for\s+)?)\s+(?P<target>[a-z][a-z .'-]{1,60}?)\s*[?.!]*$", re.I)
TOPIC_WORDS = set(
    "the a an iit ropar iitrpr rupnagar institute campus college hostel hostels mess fest fests zeitgeist advitiya "
    "aarohan club clubs board boards council department dept director dean registrar hod head warden wardens "
    "library librarian placement placements cell office faculty professor professors staff course courses "
    "admission admissions exam exams semester convocation centre center lab labs program programme".split())


def is_private_person(q: str, tables: TableStore) -> bool:
    """A profile request about a named person who isn't official faculty/staff or a listed office-bearer.
    We answer only from official data for these and never search the web for them."""
    m = PROFILE.search(q)
    if not m or tables.find_people(q):
        return False
    target = re.sub(r"\b(dr|prof|mr|ms|mrs|sh|smt)\b\.?", " ", m.group("target").lower())
    words = re.findall(r"[a-z]+", target)
    if not 1 <= len(words) <= 4 or any(w in TOPIC_WORDS for w in words):
        return False
    return not (tables.find_departments(target) or tables.find_committees(target))


def is_student_lookup(q: str, tables: TableStore) -> bool:
    ql = q.lower()
    if ENTRY_NO.search(ql):
        return True
    if tables.find_people(q):  # official faculty/staff/office-bearers are fine to look up
        return False
    # "cgpa of Rahul", "phone number of Riya from CSE"
    m = re.search(rf"\b{ATTR}\s+(?:no\.?\s+|number\s+)?of\s+(.{{0,40}})", ql)
    if m and not OFFICIAL.search(m.group(2)):
        return True
    # "Rahul's room number", "her instagram"
    if re.search(rf"\b(\w+'s|his|her)\s+{ATTR}\b", ql) and not OFFICIAL.search(ql):
        return True
    # "phone numbers of CSE students", "instagram of a junior"
    if re.search(rf"\b{STRICT_ATTR}\b.{{0,40}}\b{STUDENT}\b|\b{STUDENT}\b.{{0,40}}\b{STRICT_ATTR}\b", ql):
        return True
    # "list of students in the 2023 batch" (but office-bearers and statistics are fine)
    if re.search(rf"\b(list|names|name|all|find|who are|who all|details)\b.{{0,30}}\b{STUDENT}\b", ql) \
            and not OFFICE_BEARER.search(ql):
        return True
    return False


# --- Explicit web search -------------------------------------------------------------------
WEB_ASK = re.compile(
    r"\b(search|look\s+(?:it\s+|this\s+|that\s+)?up|google|check|find)\b[^.?!]{0,30}\b(web|internet|online|google)\b"
    r"|\bweb\s*search\b|\bgoogle\s+(?:it|this|that)\b|\bsearch\s+online\b"
    r"|\bsearch\b[^.?!]{0,30}\b(beyond|outside)\b", re.I)
WIDE = re.compile(r"\b(whole|entire|wider|wide|open|full|any|other|all)\s+(web|internet|sites?|websites?)\b"
                  r"|\beverywhere\b|\b(beyond|outside)\s+(?:the\s+)?official\b", re.I)
WEB_PHRASES = re.compile(
    r"\b(?:please|pls|kindly|can you|could you|would you|now|also|again|instead)\b"
    r"|\b(?:search|look(?:\s+(?:it|this|that))?\s+up|google|check|find)\b"
    r"|\b(?:(?:on|in|across|from|beyond|outside)\s+)?(?:the\s+)?(?:(?:whole|entire|wider|wide|open|full)\s+)?"
    r"(?:web|internet|online|official\s+(?:sites?|websites?))\b"
    r"|\bweb\s*search\b", re.I)
LEADING_FILLER = re.compile(r"^(?:(?:for|about|it|this|that|the|on)\b[\s:,-]*)+", re.I)


def web_request(q: str) -> str | None:
    """'official' or 'open' if the user explicitly asks to search the web, else None."""
    if not WEB_ASK.search(q):
        return None
    return "open" if WIDE.search(q) else "official"


def strip_web_phrase(q: str) -> str:
    """'Please search the web for Zeitgeist 2026 dates' -> 'Zeitgeist 2026 dates' ('' if nothing is left)."""
    core = re.sub(r"\s{2,}", " ", WEB_PHRASES.sub(" ", q)).strip(" ,.:;?!-")
    return LEADING_FILLER.sub("", core).strip(" ,.:;?!-")


# --- Live (time-sensitive) ----------------------------------------------------------------
LIVE = re.compile(
    r"\b(latest|recent|recently|upcoming|current|currently open|ongoing|deadline|deadlines|last date|"
    r"notice|notices|announcement|announcements|circular|circulars|news|holiday|holidays|vacanc\w*|"
    r"recruitment|hiring|job openings?|openings|this week|this month|today|tomorrow|next holiday|"
    r"still open|accepting applications|applications? open|apply by)\b", re.I)

# --- Table lookups -----------------------------------------------------------------------
PERSON_WORDS = (r"(professors?|profs?|faculty|faculties|teachers?|staff|members|wardens?|hods?|head of department|"
                r"doctors?|nurses?|librarians?|counsell?ors?|counselors?|caretakers?)")
# "reach" is left out on purpose: "how do I reach IIT Ropar" is about travel, not a contact lookup
CONTACT_WORDS = r"(email|e-mail|mail id|mail|phone|contact|extension|office|room)"
TABLE_PATTERNS = [
    re.compile(rf"\b(list|all|names?)\b.{{0,30}}\b{PERSON_WORDS}\b", re.I),
    re.compile(rf"\b{PERSON_WORDS}\b\s+(in|of|from|at|working|who|for|on|doing|with)\b", re.I),
    re.compile(r"\bwho\s+(teaches|teach|works on|work on|is working on|researches|does research)\b", re.I),
    re.compile(rf"\b{CONTACT_WORDS}\b.{{0,15}}\b(of|for)\b", re.I),
    re.compile(rf"\b(how (do|can) i|how to)\s+(contact|email|mail)\b", re.I),
    re.compile(rf"\bwho\s+(is|are)\s+the\s+{PERSON_WORDS}\b", re.I),
]


@dataclass
class Route:
    kind: str                  # refuse | live | table | retrieval
    reason: str
    entities: dict = field(default_factory=dict)


def route(question: str, tables: TableStore) -> Route:
    if is_student_lookup(question, tables):
        return Route("refuse", "asks for an individual student's details")

    if m := LIVE.search(question):
        return Route("live", f"time-sensitive word '{m.group(0)}'")

    depts, committees = tables.find_departments(question), tables.find_committees(question)
    people, roles = tables.find_people(question), tables.find_roles(question)
    entities = {k: v for k, v in [("departments", depts), ("committees", committees), ("people", people),
                                  ("roles", roles)] if v}
    if people:
        return Route("table", "names a faculty/staff member or office-bearer", entities)
    if roles:
        return Route("table", f"asks for a role ({roles[0]})", entities)
    ql = question.lower()
    if committees and re.search(rf"\b(who|members?|contact|email|{CONTACT_WORDS}|advisor|secretary)\b", ql):
        return Route("table", "asks who is on a board/committee", entities)
    for p in TABLE_PATTERNS:
        if m := p.search(question):
            return Route("table", f"lookup phrase '{m.group(0).strip()}'", entities)
    if depts and re.search(rf"\b{PERSON_WORDS}\b", ql):
        return Route("table", "department + people", entities)
    return Route("retrieval", "general question", entities)

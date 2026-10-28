import json

import pytest

from app import config
from app.live import LiveAPI
from app.router import route
from app.tables import TableStore


@pytest.fixture(scope="module")
def tables():
    return TableStore()


@pytest.mark.parametrize("question, kind", [
    # the four "Done" questions
    ("Who teaches machine learning in CSE?", "table"),
    ("When is Zeitgeist?", "retrieval"),
    ("What are the hostel rules?", "retrieval"),
    ("Latest PhD admission notices?", "live"),
    # lookups
    ("CSE professors", "table"),
    ("who is the BoCA faculty advisor", "table"),
    ("email of the hostel office", "table"),
    ("Dr. Puneet Goyal email", "table"),
    ("who is the general secretary of the cultural board", "table"),
    ("list of faculty in mechanical engineering", "table"),
    ("how do I contact the medical centre", "table"),
    # time-sensitive
    ("PhD deadlines", "live"),
    ("upcoming holidays", "live"),
    ("any job openings?", "live"),
    ("latest notices", "live"),
    # general
    ("what are the clubs at IIT Ropar", "retrieval"),
    ("What is the minimum CGPA for branch change?", "retrieval"),
    ("What is the grading system for students?", "retrieval"),
    ("How many students are in the CSE batch?", "retrieval"),
    ("What does BoCA do?", "retrieval"),
])
def test_routes(tables, question, kind):
    assert route(question, tables).kind == kind


@pytest.mark.parametrize("question", [
    "What is the phone number of Rahul from CSE?",
    "list of students in 2023 batch",
    "who is 2023CSB1234",
    "what is Riya's room number",
    "give me instagram ids of freshers",
    "cgpa of Ankit Sharma",
    "names of all students in mechanical",
])
def test_student_lookups_are_refused(tables, question):
    assert route(question, tables).kind == "refuse"


@pytest.mark.parametrize("question", [
    "who are the student council members",
    "who is the general secretary of BoST",
    "phone number of the hostel office",
    "address of IIT Ropar",
])
def test_office_bearers_and_offices_are_allowed(tables, question):
    assert route(question, tables).kind != "refuse"


def test_boca_faculty_advisor(tables):
    text = "\n".join(s.text for s in tables.lookup("who is the BoCA faculty advisor"))
    assert "Anupam Bandyopadhyay" in text and "boca@iitrpr.ac.in" in text


def test_cse_machine_learning_faculty(tables):
    text = "\n".join(s.text for s in tables.lookup("Who teaches machine learning in CSE?"))
    assert "Basant Subba" in text and "Mudasir Ahmad" in text
    assert "Physics" not in text


def test_hostel_office_email(tables):
    text = "\n".join(s.text for s in tables.lookup("email of the hostel office"))
    assert "carehostel@iitrpr.ac.in" in text


def _saved_api(path: str):
    """Offline stand-in for the live API, using the scraper's saved responses."""
    name = {"get-public-global-news": "news", "admissions/active?status=1": "admissions_active",
            "v2/public/advertisements?status=1": "v2_public_advertisements"}.get(path, "calendar_events")
    items = json.loads((config.ROOT / "data/raw/api" / f"{name}.json").read_text(encoding="utf-8"))
    key = {"news": "news", "admissions_active": "admissions", "v2_public_advertisements": "advertisements",
           "calendar_events": "events"}[name]
    return {key: items}


def test_live_phd_notices_only_admissions():
    sources = LiveAPI(fetch=_saved_api).search("Latest PhD admission notices?")
    assert sources and all(s.source_type == "live" for s in sources)
    assert all("phd" in s.title.lower() or "ph.d" in s.title.lower() for s in sources)


def test_live_jobs():
    sources = LiveAPI(fetch=_saved_api).search("any job openings?")
    assert sources and all(s.title.startswith("Job/Position") for s in sources)


@pytest.mark.parametrize("question", [
    "tell all you know about shlok vaidya",
    "Tell me everything about Riya Sharma",
    "who is Aman Gupta?",
    "details of Priya",
    "information about Rohan Mehta",
])
def test_private_person_profiles_are_detected(tables, question):
    from app.router import is_private_person
    assert is_private_person(question, tables)


@pytest.mark.parametrize("question", [
    "who is Dr. Puneet Goyal?",          # official faculty
    "tell me about Anupam Bandyopadhyay",  # official faculty advisor
    "who is the director?",
    "tell me about zeitgeist",
    "tell me about BoCA",
    "who is the warden of Beas hostel",
    "information about the hostel fee",
    "What are the hostel rules?",
])
def test_official_people_and_topics_are_not_private(tables, question):
    from app.router import is_private_person
    assert not is_private_person(question, tables)

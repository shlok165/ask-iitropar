from app.index import load_chunks
from app.search import tokenize


def test_tokenize_keeps_course_codes_and_acronyms():
    assert tokenize("Who teaches CS201 in the BoCA?") == ["teaches", "cs201", "boca"]


def test_tokenize_drops_punctuation():
    assert tokenize("hostel-rules, 2025!") == ["hostel", "rules", "2025"]


def test_chunk_ids_are_unique_and_nothing_dropped():
    chunks = load_chunks()
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(chunks) == 7909


def test_chunks_carry_metadata_for_the_prompt():
    c = load_chunks()[0]
    assert c["source_type"] in {"site", "pdf", "api"}
    assert c["url"].startswith("http") and c["fetched"]

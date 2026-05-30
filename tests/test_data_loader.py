# ── BC5CDR loader tests ────────────────────────────────────────────────────────

from src.data.bc5cdr_loader import BC5CDRLoader, NERExample, BC5CDR_TO_CANONICAL


def test_bc5cdr_loads_all_splits():
    """BC5CDR loads all three splits without error."""
    loader = BC5CDRLoader()
    train, val, test = loader.load()
    assert len(train) == 5228
    assert len(val)   == 5330
    assert len(test)  == 5865


def test_bc5cdr_example_structure():
    """Each example has equal-length tokens and labels."""
    loader = BC5CDRLoader()
    train, _, _ = loader.load()
    for ex in train[:20]:
        assert isinstance(ex, NERExample)
        assert len(ex.tokens) == len(ex.labels)
        assert len(ex.tokens) > 0


def test_bc5cdr_canonical_labels_only():
    """No raw BC5CDR label strings remain after conversion."""
    loader      = BC5CDRLoader()
    train, _, _ = loader.load()
    valid       = set(BC5CDR_TO_CANONICAL.values())
    for ex in train[:100]:
        for label in ex.labels:
            assert label in valid, (
                f"Non-canonical label found: '{label}' in {ex.example_id}"
            )


def test_bc5cdr_entity_spans():
    """entity_spans extracts correctly from a known example."""
    ex = NERExample(
        example_id = "test_0",
        tokens     = ["Naloxone", "reverses", "effects", "of", "morphine"],
        labels     = ["B-TREATMENT", "O", "O", "O", "B-TREATMENT"]
    )
    spans = ex.entity_spans
    assert len(spans) == 2
    assert ("Naloxone", "TREATMENT") in spans
    assert ("morphine", "TREATMENT") in spans


def test_bc5cdr_multiword_span():
    """Multi-word entity spans are extracted correctly."""
    ex = NERExample(
        example_id = "test_1",
        tokens     = ["alpha", "-", "methyldopa", "causes", "hypotension"],
        labels     = ["B-TREATMENT", "I-TREATMENT", "I-TREATMENT",
                      "O", "B-PROBLEM"]
    )
    spans = ex.entity_spans
    assert ("alpha - methyldopa", "TREATMENT") in spans
    assert ("hypotension", "PROBLEM") in spans


def test_bc5cdr_stats_keys():
    """get_stats returns expected keys and positive values."""
    loader      = BC5CDRLoader()
    train, _, _ = loader.load()
    stats       = loader.get_stats(train)
    assert stats["num_examples"]   == 5228
    assert stats["total_entities"] > 0
    assert "TREATMENT" in stats["entities_by_type"]
    assert "PROBLEM"   in stats["entities_by_type"]
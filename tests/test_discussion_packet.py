"""Unit tests for Discussion Evidence Packet generation.

Covers quality gate logic, evidence normalisation, lineage preservation,
data-support mapping, and the priority-tier assignment.  All tests run
without an LLM or ChromaDB connection.
"""


from src.core.discussion_packet import (
    SCHEMA_VERSION,
    _build_candidate,
    _collect_limitations,
    _data_support,
    _hypothesis_claim,
    _is_citable,
    _normalize_evidence,
    _priority_tier,
    _split_site,
    build_discussion_evidence_packet,
)
from src.core.models import (
    CoScientistState,
    ExperimentDesign,
    Hypothesis,
)

# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_hypothesis(
    *,
    condition: str = "IF EGFR is activated",
    prediction: str = "THEN SRC-Y416 phosphorylation increases",
    mechanism: str = "BECAUSE EGFR activates SRC kinase",
    supporting_ptms: list | None = None,
    signaling_chain: str = "EGFR → SRC → VIM-S56",
    testable_prediction: str = "Inhibit EGFR and measure SRC-Y416 by Western Blot",
    evidence_for: list | None = None,
    evidence_against: list | None = None,
    debate_history: list | None = None,
    parent_hypothesis_ids: list | None = None,
    evolution_type: str = "",
    addressed_critiques: list | None = None,
    elo_rating: int = 1600,
) -> Hypothesis:
    return Hypothesis(
        condition=condition,
        prediction=prediction,
        mechanism=mechanism,
        supporting_ptms=supporting_ptms or ["SRC-Y416", "EGFR-Y1068"],
        signaling_chain=signaling_chain,
        testable_prediction=testable_prediction,
        evidence_for=evidence_for or [],
        evidence_against=evidence_against or [],
        debate_history=debate_history or [],
        parent_hypothesis_ids=parent_hypothesis_ids or [],
        evolution_type=evolution_type,
        addressed_critiques=addressed_critiques or [],
        elo_rating=elo_rating,
    )


def _citable_evidence(stance: str = "supporting") -> dict:
    return {
        "evidence_id": "ev-001",
        "title": "EGFR-SRC signalling in cancer",
        "pmid": "12345678",
        "doi": "10.1000/test",
        "authors": "Smith J et al.",
        "year": "2020",
        "journal": "Nature",
        "collection": "ptm_articles",
        "retrieval_score": 0.92,
        "excerpt": "EGFR activates SRC through direct phosphorylation...",
        "stance": stance,
    }


def _enriched_ptm_record(gene: str = "SRC", position: str = "Y416") -> dict:
    return {
        "gene": gene,
        "position": position,
        "condition": "EGF_5min",
        "ptm_relative_log2fc": 1.8,
        "protein_log2fc": 0.2,
        "pathways": ["EGFR signaling", "Focal adhesion"],
    }


def _minimal_state(hypothesis: Hypothesis | None = None) -> CoScientistState:
    state = CoScientistState(
        research_goal="Identify novel EGFR-SRC therapeutic targets",
        experimental_context={"ptm_type": "phosphorylation"},
        enriched_ptm_data=[_enriched_ptm_record()],
        rag_collections=["ptm_articles"],
    )
    if hypothesis:
        state.hypotheses = [hypothesis]
    return state


# ─── _split_site ────────────────────────────────────────────────────────────

def test_split_site_standard():
    gene, position = _split_site("SRC-Y416")
    assert gene == "SRC"
    assert position == "Y416"


def test_split_site_no_dash():
    gene, position = _split_site("EGFR")
    assert gene == "EGFR"
    assert position == ""


def test_split_site_empty():
    gene, position = _split_site("")
    assert gene == ""
    assert position == ""


# ─── _hypothesis_claim ──────────────────────────────────────────────────────

def test_hypothesis_claim_full():
    h = _make_hypothesis()
    claim = _hypothesis_claim(h)
    assert claim == {
        "if": h.condition,
        "then": h.prediction,
        "because": h.mechanism,
    }


def test_hypothesis_claim_partial():
    h = _make_hypothesis(mechanism="")
    claim = _hypothesis_claim(h)
    assert claim["if"] == h.condition
    assert claim["then"] == h.prediction
    assert claim["because"] == ""


# ─── _is_citable ────────────────────────────────────────────────────────────

def test_is_citable_with_pmid():
    assert _is_citable({"title": "A paper", "pmid": "123"})


def test_is_citable_with_doi():
    assert _is_citable({"title": "A paper", "doi": "10.1/x"})


def test_is_citable_with_evidence_id_only():
    assert _is_citable({"title": "A paper", "evidence_id": "ev-01"})


def test_is_citable_missing_title():
    assert not _is_citable({"pmid": "123"})


def test_is_citable_missing_identifier():
    assert not _is_citable({"title": "A paper"})


# ─── _normalize_evidence ────────────────────────────────────────────────────

def test_normalize_evidence_deduplication():
    entry = _citable_evidence()
    normalized = _normalize_evidence([entry, entry], "supporting")
    assert len(normalized) == 1


def test_normalize_evidence_preserves_fields():
    entry = _citable_evidence()
    normalized = _normalize_evidence([entry], "supporting")
    assert normalized[0]["pmid"] == "12345678"
    assert normalized[0]["doi"] == "10.1000/test"
    assert normalized[0]["retrieval_score"] == 0.92
    assert normalized[0]["stance"] == "supporting"


def test_normalize_evidence_legacy_source_key():
    legacy = {"source": "Old paper", "text": "Some text"}
    normalized = _normalize_evidence([legacy], "supporting")
    assert normalized[0]["title"] == "Old paper"
    assert normalized[0]["excerpt"] == "Some text"


# ─── _data_support ──────────────────────────────────────────────────────────

def test_data_support_match():
    records = [_enriched_ptm_record("SRC", "Y416")]
    support = _data_support(["SRC-Y416"], records)
    assert len(support) == 1
    assert support[0]["gene"] == "SRC"
    assert support[0]["ptm_relative_log2fc"] == 1.8


def test_data_support_no_match():
    records = [_enriched_ptm_record("EGFR", "Y1068")]
    support = _data_support(["SRC-Y416"], records)
    assert support == []


def test_data_support_gene_only():
    records = [_enriched_ptm_record("SRC", "Y416"), _enriched_ptm_record("SRC", "Y527")]
    support = _data_support(["SRC"], records)
    assert len(support) == 2


# ─── _collect_limitations ───────────────────────────────────────────────────

def test_collect_limitations_from_debate():
    h = _make_hypothesis(debate_history=[{"critique": "Lacks temporal evidence"}])
    limitations = _collect_limitations(h, [])
    assert "Lacks temporal evidence" in limitations


def test_collect_limitations_from_counter_evidence():
    h = _make_hypothesis()
    counter = [{"title": "Contradicting paper", "excerpt": "SRC is not activated by EGFR here"}]
    limitations = _collect_limitations(h, counter)
    assert any("Contradicting paper" in lim for lim in limitations)


def test_collect_limitations_deduplication():
    h = _make_hypothesis(debate_history=[
        {"critique": "Same critique"},
        {"critique": "Same critique"},
    ])
    limitations = _collect_limitations(h, [])
    assert limitations.count("Same critique") == 1


# ─── _priority_tier ─────────────────────────────────────────────────────────

def test_priority_tier_values():
    assert _priority_tier(0) == "high"
    assert _priority_tier(1) == "medium"
    assert _priority_tier(2) == "exploratory"
    assert _priority_tier(99) == "exploratory"


# ─── Quality gate ───────────────────────────────────────────────────────────

def test_quality_gate_passes_full_hypothesis():
    h = _make_hypothesis(
        evidence_for=[_citable_evidence("supporting")],
        evidence_against=[_citable_evidence("contradicting")],
    )
    state = _minimal_state(h)
    _, gate = _build_candidate(h, state)
    assert gate["passed"], f"Expected gate to pass, got: {gate['reason']}"


def test_quality_gate_fails_missing_claim():
    h = _make_hypothesis(condition="", prediction="", mechanism="")
    state = _minimal_state(h)
    _, gate = _build_candidate(h, state)
    assert not gate["passed"]
    assert "missing_if_then_because_claim" in gate["reason"]


def test_quality_gate_fails_no_data_support():
    h = _make_hypothesis(supporting_ptms=["UNKNOWN-X999"])
    state = _minimal_state(h)
    _, gate = _build_candidate(h, state)
    assert not gate["passed"]
    assert "missing_measured_ptm_data_support" in gate["reason"]


def test_quality_gate_fails_no_citable_literature():
    h = _make_hypothesis(
        evidence_for=[{"title": "", "pmid": ""}],
        evidence_against=[{"title": "Counter", "pmid": "999"}],
    )
    state = _minimal_state(h)
    _, gate = _build_candidate(h, state)
    assert not gate["passed"]
    assert "missing_citable_supporting_literature" in gate["reason"]


def test_quality_gate_fails_no_uncertainty():
    h = _make_hypothesis(
        evidence_for=[_citable_evidence()],
        evidence_against=[],
        debate_history=[],
    )
    state = _minimal_state(h)
    _, gate = _build_candidate(h, state)
    assert not gate["passed"]
    assert "missing_counter_evidence_or_debate_limitation" in gate["reason"]


# ─── build_discussion_evidence_packet ───────────────────────────────────────

def _full_hypothesis() -> Hypothesis:
    return _make_hypothesis(
        evidence_for=[_citable_evidence("supporting")],
        evidence_against=[_citable_evidence("contradicting")],
        debate_history=[{"critique": "Temporal context not addressed"}],
        parent_hypothesis_ids=["parent-01"],
        evolution_type="strengthened",
        addressed_critiques=["Temporal context not addressed"],
    )


def test_packet_schema_version():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert packet["schema_version"] == SCHEMA_VERSION


def test_packet_contains_usage_notice():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert "usage_notice" in packet
    assert len(packet["usage_notice"]) > 20


def test_packet_eligible_hypothesis_included():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert len(packet["selected_hypotheses"]) == 1
    assert packet["status"] == "ready"


def test_packet_ineligible_hypothesis_excluded():
    h = _make_hypothesis(condition="", prediction="", mechanism="")
    state = _minimal_state(h)
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert packet["selected_hypotheses"] == []
    assert packet["status"] == "no_eligible_hypotheses"
    assert len(packet["quality_summary"]["excluded_candidates"]) == 1


def test_packet_max_hypotheses_respected():
    hypotheses = [_full_hypothesis() for _ in range(5)]
    state = CoScientistState(
        research_goal="test",
        experimental_context={"ptm_type": "phosphorylation"},
        enriched_ptm_data=[_enriched_ptm_record()],
        rag_collections=[],
        hypotheses=hypotheses,
    )
    packet = build_discussion_evidence_packet(state, session_id="test-01", max_hypotheses=2)
    assert len(packet["selected_hypotheses"]) <= 2


def test_packet_lineage_preserved():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    lineage = packet["selected_hypotheses"][0]["lineage"]
    assert lineage["parent_hypothesis_ids"] == ["parent-01"]
    assert lineage["evolution_type"] == "strengthened"
    assert "Temporal context not addressed" in lineage["addressed_critiques"]


def test_packet_elo_not_exported():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    hyp = packet["selected_hypotheses"][0]
    assert "elo_rating" not in hyp


def test_packet_claim_is_structured_if_then_because():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    claim = packet["selected_hypotheses"][0]["claim"]
    assert claim["if"]
    assert claim["then"]
    assert claim["because"]


def test_packet_priority_tier_assigned():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert packet["selected_hypotheses"][0]["priority_tier"] == "high"


def test_packet_experiment_priorities_linked():
    h = _full_hypothesis()
    design = ExperimentDesign(
        hypothesis_id=h.id,
        title="Western Blot for SRC-Y416",
        objective="Validate SRC activation",
        approach="Western Blot",
        controls=["DMSO control"],
        expected_outcome="Increased SRC-Y416 band",
        alternative_outcome="No change",
    )
    state = CoScientistState(
        research_goal="test",
        experimental_context={"ptm_type": "phosphorylation"},
        enriched_ptm_data=[_enriched_ptm_record()],
        rag_collections=[],
        hypotheses=[h],
        experiment_designs=[design],
    )
    packet = build_discussion_evidence_packet(state, session_id="test-01")
    assert len(packet["experiment_priorities"]) == 1
    assert packet["experiment_priorities"][0]["hypothesis_id"] == h.id


def test_packet_source_orders_propagated():
    state = _minimal_state(_full_hypothesis())
    packet = build_discussion_evidence_packet(
        state, session_id="test-01", source_orders=["ORDER_001", "ORDER_002"]
    )
    assert packet["source_orders"] == ["ORDER_001", "ORDER_002"]

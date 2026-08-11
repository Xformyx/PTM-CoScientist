"""Unit tests for Evidence Graph, Reflection, Proximity, and pipeline integration."""

from __future__ import annotations

import json

from src.agents.proximity import cluster_and_select_diverse_hypotheses
from src.agents.reflection import run_reflection
from src.core.evidence_graph import build_evidence_graph, graph_neighborhood
from src.core.models import CoScientistState, Hypothesis, HypothesisStatus, LabResult
from src.core.pipeline import CoScientistPipeline


class FakeLLM:
    """Deterministic LLM double that emits schema-valid outputs by agent role."""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        if "hypothesis generator" in system_prompt.lower():
            return json.dumps([
                {
                    "condition": "IF treatment activates EGFR",
                    "prediction": "THEN SRC-Y416 increases",
                    "mechanism": "BECAUSE EGFR activates SRC",
                    "category": "mechanistic",
                    "supporting_ptms": ["SRC-Y416"],
                    "signaling_chain": "EGFR → SRC → VIM-S56",
                    "testable_prediction": "Inhibit EGFR and measure SRC-Y416",
                },
                {
                    "condition": "IF treatment activates EGFR",
                    "prediction": "THEN STAT3-Y705 increases",
                    "mechanism": "BECAUSE EGFR signals through JAK",
                    "category": "temporal",
                    "supporting_ptms": ["STAT3-Y705"],
                    "signaling_chain": "EGFR → JAK → STAT3-Y705",
                    "testable_prediction": "Inhibit JAK and measure STAT3-Y705",
                },
            ])
        if "performing self-critique" in system_prompt.lower():
            literature_status = "supported" if "EGFR-SRC paper" in prompt or "99999999" in prompt else "insufficient"
            return json.dumps({
                "atomic_claims": [{"claim": "EGFR activates SRC", "evidence_status": "untested"}],
                "data_consistency": "partial",
                "literature_consistency": literature_status,
                "novelty_assessment": "uncertain",
                "confounders": ["Total protein abundance may change"],
                "missing_evidence": ["Direct kinase assay"],
                "falsification_conditions": ["SRC-Y416 remains unchanged after EGFR inhibition"],
                "recommended_action": "advance",
                "summary": "Candidate is testable but needs direct validation.",
            })
        if "critical scientific reviewer" in system_prompt.lower():
            return json.dumps({
                "winner": "A",
                "reasoning": "A has direct PTM support.",
                "critique_a": "Needs inhibitor control.",
                "critique_b": "Temporal evidence is incomplete.",
            })
        if "literature evidence classifier" in system_prompt.lower():
            return "SUPPORTING"
        if "evolution specialist" in system_prompt.lower():
            return json.dumps([])
        return "NEUTRAL"


class FakeChroma:
    def is_available(self):
        return False

    def search_for_hypothesis(self, *args, **kwargs):
        return [{
            "id": "doc-src-1",
            "document": "EGFR can activate SRC in epithelial models.",
            "collection": "ptm_articles",
            "distance": 0.2,
            "metadata": {
                "title": "EGFR-SRC paper",
                "pmid": "99999999",
                "doi": "10.1000/egfr-src",
            },
        }]


class FakePTMConnector:
    def assemble_context(self, order_code: str, ptm_type: str):
        return _context()

    def assemble_multi_context(self, order_codes: list[str], ptm_type: str):
        return _context()


def _context():
    return {
        "order_code": "ORDER_001",
        "ptm_type": "phosphorylation",
        "enriched_ptm_count": 2,
        "top_ptms": [
            {
                "gene": "SRC",
                "position": "Y416",
                "ptm_relative_log2fc": 1.5,
                "protein_log2fc": 0.1,
                "pathways": ["EGFR signaling"],
            },
            {
                "gene": "STAT3",
                "position": "Y705",
                "ptm_relative_log2fc": 1.2,
                "protein_log2fc": 0.0,
                "pathways": ["JAK-STAT"],
            },
        ],
        "kinase_modules": {
            "kinase_modules": [{"kinase": "SRC", "substrates": ["VIM-S56"]}],
        },
        "signal_flow": {"edges": [{"source": "EGFR", "target": "SRC", "relation": "ACTIVATES"}]},
        "comovement_clusters": {"clusters": {"early": ["SRC-Y416", "STAT3-Y705"]}},
    }


def _hypothesis(site: str = "SRC-Y416", elo: int = 1500) -> Hypothesis:
    return Hypothesis(
        condition="IF EGFR is activated",
        prediction=f"THEN {site} increases",
        mechanism="BECAUSE EGFR activates downstream signalling",
        supporting_ptms=[site],
        signaling_chain=f"EGFR → SRC → {site}",
        testable_prediction=f"Measure {site} after EGFR inhibition",
        elo_rating=elo,
    )


def test_evidence_graph_builds_measurement_regulator_signal_and_lab_edges():
    hypothesis = _hypothesis()
    hypothesis.evidence_for = [{"evidence_id": "doc-1", "title": "SRC paper", "pmid": "123"}]
    lab_result = LabResult(hypothesis_id=hypothesis.id, outcome="supports", assay_type="Western blot")
    graph = build_evidence_graph(_context(), [hypothesis], [lab_result])

    node_ids = {node["id"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    assert "ptm_site:SRC-Y416" in node_ids
    assert f"hypothesis:{hypothesis.id}" in node_ids
    assert "MODIFIES" in relations
    assert "ACTIVATES" in relations
    assert "SUPPORTS" in relations
    assert graph["summary"]["node_count"] > 0


def test_evidence_graph_neighborhood_is_bounded_and_relevant():
    graph = build_evidence_graph(_context())
    neighbors = graph_neighborhood(graph, ["SRC-Y416"], max_edges=2)
    assert len(neighbors) <= 2
    assert neighbors


def test_reflection_attaches_structured_self_critique():
    hypothesis = _hypothesis()
    graph = build_evidence_graph(_context(), [hypothesis])
    reviewed = run_reflection(
        [hypothesis],
        context={**_context(), "research_goal": "Test EGFR-SRC"},
        evidence_graph=graph,
        lab_results=[],
        llm=FakeLLM(),
    )
    reflection = reviewed[0].reflection
    assert reviewed[0].status == HypothesisStatus.REFLECTED
    assert reflection["recommended_action"] == "advance"
    assert "Total protein abundance may change" in reflection["confounders"]


def test_proximity_selects_cluster_representatives_before_duplicates():
    first = _hypothesis("SRC-Y416", 1700)
    duplicate = _hypothesis("SRC-Y416", 1600)
    diverse = _hypothesis("STAT3-Y705", 1500)
    selected, summary = cluster_and_select_diverse_hypotheses(
        [first, duplicate, diverse], max_hypotheses=2
    )
    assert selected[0].id == first.id
    assert diverse.id in {candidate.id for candidate in selected}
    assert summary["cluster_count"] >= 2
    assert first.proximity_cluster


def test_pipeline_integrates_reflection_graph_and_diversity():
    pipeline = CoScientistPipeline(
        llm=FakeLLM(),
        chromadb=FakeChroma(),
        ptm_connector=FakePTMConnector(),
        max_iterations=1,
        generate_candidates=2,
        tournament_rounds=1,
        evolve_top_k=1,
        reflection_enabled=True,
        evidence_graph_enabled=True,
        proximity_enabled=True,
        max_diverse_hypotheses=2,
    )
    state = pipeline.run(order_code="ORDER_001", research_goal="Test EGFR signalling")
    assert len(state.hypotheses) == 2
    assert all(hypothesis.reflection for hypothesis in state.hypotheses)
    assert all(hypothesis.evidence_for for hypothesis in state.hypotheses)
    assert all(
        hypothesis.reflection.get("literature_consistency") == "supported"
        for hypothesis in state.hypotheses
    )
    assert state.evidence_graph["summary"]["node_count"] > 0
    assert state.diversity_summary["recommended_hypothesis_ids"]


def test_discussion_packet_includes_reasoning_provenance():
    from src.core.discussion_packet import build_discussion_evidence_packet

    hypothesis = _hypothesis()
    hypothesis.evidence_for = [{
        "evidence_id": "doc-1",
        "title": "SRC paper",
        "pmid": "123",
        "excerpt": "Relevant support.",
    }]
    hypothesis.debate_history = [{"critique": "Needs inhibitor control"}]
    hypothesis.reflection = {"recommended_action": "advance", "confounders": ["Total protein change"]}

    state = CoScientistState(
        research_goal="Test EGFR-SRC",
        enriched_ptm_data=_context()["top_ptms"],
        hypotheses=[hypothesis],
        lab_results=[LabResult(hypothesis_id=hypothesis.id, outcome="inconclusive", assay_type="Western blot")],
    )
    state.evidence_graph = build_evidence_graph(_context(), [hypothesis], state.lab_results)
    state.diversity_summary = {"recommended_hypothesis_ids": [hypothesis.id], "cluster_count": 1}
    state.meta_review = {"executive_summary": "Candidate needs validation."}

    packet = build_discussion_evidence_packet(state, session_id="reasoning-packet")
    candidate = packet["selected_hypotheses"][0]
    assert candidate["reflection"]["recommended_action"] == "advance"
    assert candidate["experimental_evidence"][0]["outcome"] == "inconclusive"
    assert packet["evidence_graph_summary"]["node_count"] > 0
    assert packet["meta_review"]["executive_summary"]

"""Tests for Meta-review, Experiment Designer evidence compatibility, and lab-result API."""

from __future__ import annotations

import json
import os

os.environ.setdefault("COSCIENTIST_SESSION_FILE", "/tmp/ptm-coscientist-reasoning-test-sessions.json")

from fastapi.testclient import TestClient

from src.agents.experiment_designer import _build_prompt
from src.agents.meta_reviewer import run_meta_review
from src.api import server
from src.core.models import CoScientistState, ExperimentDesign, Hypothesis


class MetaLLM:
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return json.dumps({
            "executive_summary": "Two competing candidates require experimental discrimination.",
            "leading_mechanism": {"hypothesis_id": "hyp-01", "rationale": "Measured site support"},
            "alternative_models": [{"hypothesis_id": "hyp-02", "rationale": "Temporal alternative"}],
            "key_uncertainties": ["Directness of kinase-substrate relation"],
            "next_best_experiment": {
                "hypothesis_id": "hyp-01",
                "rationale": "Discriminates kinase dependency",
                "discriminates_between": ["hyp-01", "hyp-02"],
            },
            "researcher_questions": ["Is an EGFR inhibitor available in this model?"],
            "usage_notice": "Interpretive candidates require review.",
        })


def _hypothesis(hypothesis_id: str = "hyp-01") -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        condition="IF EGFR is activated",
        prediction="THEN SRC-Y416 increases",
        mechanism="BECAUSE EGFR activates SRC",
        supporting_ptms=["SRC-Y416"],
        signaling_chain="EGFR → SRC → VIM-S56",
        testable_prediction="Inhibit EGFR and measure SRC-Y416",
        evidence_for=[{
            "evidence_id": "doc-001",
            "title": "EGFR and SRC",
            "pmid": "12345678",
            "excerpt": "EGFR can activate SRC in relevant settings.",
        }],
        reflection={"confounders": ["Total protein change"], "recommended_action": "advance"},
    )


def _state() -> CoScientistState:
    hypothesis = _hypothesis()
    return CoScientistState(
        research_goal="Assess EGFR-SRC signalling",
        experimental_context={
            "ptm_type": "phosphorylation",
            "top_ptms": [{"gene": "SRC", "position": "Y416", "ptm_relative_log2fc": 1.4}],
        },
        enriched_ptm_data=[{"gene": "SRC", "position": "Y416", "ptm_relative_log2fc": 1.4}],
        hypotheses=[hypothesis],
    )


def test_experiment_prompt_uses_latest_title_excerpt_and_identifier():
    prompt = _build_prompt(_hypothesis(), {})
    assert "EGFR and SRC" in prompt
    assert "12345678" in prompt
    assert "EGFR can activate SRC" in prompt


def test_meta_review_returns_normalised_auditable_summary():
    hypothesis = _hypothesis()
    review = run_meta_review(
        research_goal="Assess EGFR-SRC signalling",
        hypotheses=[hypothesis],
        evidence_graph_summary={"node_count": 4, "edge_count": 3},
        experiment_designs=[ExperimentDesign(hypothesis_id=hypothesis.id, title="EGFR inhibition", approach="Western Blot")],
        lab_results=[],
        scientist_feedback=[],
        llm=MetaLLM(),
    )
    assert review["leading_mechanism"]["hypothesis_id"] == hypothesis.id
    assert review["reviewed_hypothesis_ids"] == [hypothesis.id]
    assert review["key_uncertainties"]


def test_lab_result_api_records_evidence_and_updates_graph(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    state = _state()
    server._sessions["lab-session"] = {
        "status": "completed",
        "state": state,
        "order_codes": ["ORDER_001"],
    }
    client = TestClient(server.app)
    response = client.post(
        "/session/lab-session/lab-results",
        json={
            "hypothesis_id": "hyp-01",
            "outcome": "supports",
            "assay_type": "Western Blot",
            "result_summary": "SRC-Y416 decreased after EGFR inhibitor.",
            "controls": ["vehicle"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "lab_result_recorded"
    assert len(state.lab_results) == 1
    assert state.evidence_graph["summary"]["node_count"] > 0


def test_lab_result_api_rejects_unknown_hypothesis(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    server._sessions["lab-session"] = {"status": "completed", "state": _state()}
    client = TestClient(server.app)
    response = client.post(
        "/session/lab-session/lab-results",
        json={"hypothesis_id": "unknown", "outcome": "supports"},
    )
    assert response.status_code == 404


def test_scientific_reasoning_endpoint_returns_reflection_and_lab_provenance(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    state = _state()
    server._sessions["reasoning-session"] = {"status": "completed", "state": state}
    client = TestClient(server.app)
    response = client.get("/session/reasoning-session/scientific-reasoning")
    assert response.status_code == 200
    body = response.json()
    assert body["hypothesis_reflections"][0]["hypothesis_id"] == "hyp-01"
    assert "evidence_graph" in body

"""API tests for the report-safe Discussion Evidence Packet endpoint."""

import os

# Must be set before importing the API module, which initialises its session store.
os.environ.setdefault("COSCIENTIST_SESSION_FILE", "/tmp/ptm-coscientist-test-sessions.json")

from fastapi.testclient import TestClient

from src.api import server
from src.core.models import CoScientistState, Hypothesis


def _eligible_state() -> CoScientistState:
    hypothesis = Hypothesis(
        condition="IF EGFR is activated",
        prediction="THEN SRC-Y416 phosphorylation increases",
        mechanism="BECAUSE EGFR activates SRC kinase",
        supporting_ptms=["SRC-Y416"],
        signaling_chain="EGFR → SRC → VIM-S56",
        testable_prediction="Inhibit EGFR and measure SRC-Y416",
        evidence_for=[{
            "evidence_id": "doc-001",
            "title": "EGFR and SRC signalling",
            "pmid": "12345678",
            "doi": "10.1000/test",
            "excerpt": "EGFR signalling can activate SRC.",
            "collection": "ptm_articles",
        }],
        debate_history=[{"critique": "Temporal validation remains necessary"}],
    )
    return CoScientistState(
        research_goal="Test EGFR-SRC signalling",
        experimental_context={"ptm_type": "phosphorylation"},
        enriched_ptm_data=[{
            "gene": "SRC",
            "position": "Y416",
            "condition": "EGF_5min",
            "ptm_relative_log2fc": 1.6,
            "protein_log2fc": 0.1,
        }],
        rag_collections=["ptm_articles"],
        hypotheses=[hypothesis],
    )


def test_discussion_packet_endpoint_for_completed_session(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    server._sessions["session-01"] = {
        "status": "completed",
        "state": _eligible_state(),
        "order_codes": ["ORDER_001"],
        "created_at": "2026-07-17T00:00:00+00:00",
    }

    client = TestClient(server.app)
    response = client.get("/session/session-01/discussion-packet?max_hypotheses=2")

    assert response.status_code == 200
    body = response.json()
    assert body["packet_type"] == "discussion_evidence_packet"
    assert body["source_orders"] == ["ORDER_001"]
    assert len(body["selected_hypotheses"]) == 1
    assert "elo_rating" not in body["selected_hypotheses"][0]


def test_discussion_packet_endpoint_uses_persisted_packet(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    persisted = {"packet_type": "discussion_evidence_packet", "status": "ready"}
    server._sessions["session-02"] = {
        "status": "completed",
        "state": None,
        "_discussion_packet": persisted,
    }

    client = TestClient(server.app)
    response = client.get("/session/session-02/discussion-packet")

    assert response.status_code == 200
    assert response.json() == persisted


def test_discussion_packet_endpoint_returns_404_for_unknown_session():
    server._sessions.clear()
    client = TestClient(server.app)
    response = client.get("/session/unknown/discussion-packet")
    assert response.status_code == 404


def test_discussion_packet_endpoint_returns_409_without_result(monkeypatch):
    monkeypatch.setattr(server, "_save_sessions", lambda: None)
    server._sessions.clear()
    server._sessions["session-running"] = {"status": "running", "state": None}
    client = TestClient(server.app)
    response = client.get("/session/session-running/discussion-packet")
    assert response.status_code == 409

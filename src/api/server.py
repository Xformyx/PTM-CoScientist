"""
FastAPI Server for PTM-CoScientist.

Provides REST API for:
- Running the Co-Scientist pipeline
- Submitting scientist feedback
- Retrieving hypothesis results and experiment designs
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import get_settings
from src.agents.experiment_designer import run_experiment_design
from src.connectors.chromadb_connector import ChromaDBConnector
from src.connectors.ptm_platform_connector import PTMPlatformConnector
from src.core.discussion_packet import build_discussion_evidence_packet
from src.core.llm_client import LLMClient
from src.core.models import CoScientistState
from src.core.pipeline import CoScientistPipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="PTM-CoScientist", version="0.1.0")

# ─── Session store — persisted to disk so restarts don't lose results ────
_SESSION_FILE = Path(
    os.getenv("COSCIENTIST_SESSION_FILE", "/data/coscientist/outputs/.sessions.json")
)
_sessions_lock = threading.Lock()


def _sessions_path() -> Path:
    """Return path to the session store file, ensuring parent dir exists."""
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    return _SESSION_FILE


def _load_sessions() -> dict:
    """Load sessions from disk. Returns empty dict on missing/corrupt file."""
    p = _sessions_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        # Mark any sessions that were 'running' at shutdown as error
        for s in raw.values():
            if s.get("status") in ("running", "cancelling"):
                s["status"] = "error"
                s["error"] = "서버가 재시작되어 파이프라인이 중단됐습니다."
        return raw
    except Exception as e:  # noqa: BLE001 - corrupt or inaccessible persisted state must not stop the API
        logger.warning(f"[Sessions] Failed to load session file: {e}")
        return {}


def _save_sessions() -> None:
    """Persist sessions (excluding unpicklable state objects) to disk."""
    p = _sessions_path()
    try:
        serialisable = {}
        for sid, s in _sessions.items():
            entry = {k: v for k, v in s.items() if k != "state"}
            # Persist top hypotheses so results survive restarts
            state: CoScientistState | None = s.get("state")
            if state and state.hypotheses:
                entry["_hypotheses"] = [h.to_dict() for h in state.hypotheses[:20]]
                entry["_iteration"] = state.iteration
                entry["_experiment_designs"] = [e.to_dict() for e in state.experiment_designs[:10]]
                entry["_discussion_packet"] = build_discussion_evidence_packet(
                    state,
                    session_id=sid,
                    source_orders=s.get("order_codes", []),
                    created_at=s.get("created_at"),
                )
            serialisable[sid] = entry
        p.write_text(json.dumps(serialisable, default=str))
    except Exception as e:  # noqa: BLE001 - persistence failures are logged without breaking active sessions
        logger.warning(f"[Sessions] Failed to save sessions: {e}")


# Boot: load any previous sessions
_sessions: dict = _load_sessions()
logger.info(f"[Sessions] Loaded {len(_sessions)} persisted session(s)")


# ─── Request/Response models ─────────────────────────────────────────────

class RunRequest(BaseModel):
    order_codes: list[str] = []   # multi-order (preferred)
    order_code: str = ""          # single-order (legacy, still supported)
    research_goal: str = ""
    ptm_type: str = "phosphorylation"
    rag_collections: list[str] | None = None
    max_iterations: int = 3
    # Per-request LLM override (empty = use server default from env)
    llm_provider: str = ""   # "" | "auto" | "ollama" | "openai" | "gemini"
    llm_model: str = ""      # e.g. "gemma3:27b", "gpt-4.1-mini", "gemini-2.5-flash"

    @property
    def resolved_order_codes(self) -> list[str]:
        if self.order_codes:
            return self.order_codes
        if self.order_code:
            return [self.order_code]
        return []


class FeedbackRequest(BaseModel):
    session_id: str
    feedback_type: str = "direction"  # direction | constraint | seed_idea
    content: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    iteration: int
    total_hypotheses: int
    top_hypotheses: list
    experiment_designs: list
    error: str | None = None


# ─── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ptm-coscientist"}


@app.get("/health/detailed")
def health_detailed() -> dict[str, Any]:
    """Deep health check: verifies ChromaDB connectivity and PTM artifacts directory."""
    settings = get_settings()
    result: dict[str, Any] = {"status": "ok", "service": "ptm-coscientist", "checks": {}}

    # ChromaDB
    try:
        chroma = ChromaDBConnector(settings.ptm_platform.chromadb_url)
        available = chroma.is_available()
        collections = chroma.list_collections() if available else []
        result["checks"]["chromadb"] = {
            "url": settings.ptm_platform.chromadb_url,
            "reachable": available,
            "collections": collections,
            "collection_count": len(collections),
        }
    except Exception as e:  # noqa: BLE001 - health checks must report connector failures, not raise them
        result["checks"]["chromadb"] = {"reachable": False, "error": str(e)}
        result["status"] = "degraded"

    # PTM artifacts directory
    artifacts_dir = Path(settings.ptm_platform.artifacts_dir)
    artifacts_ok = artifacts_dir.exists() and artifacts_dir.is_dir()
    order_dirs = sorted([p.name for p in artifacts_dir.iterdir() if p.is_dir()])[:10] if artifacts_ok else []
    result["checks"]["ptm_artifacts"] = {
        "path": str(artifacts_dir),
        "accessible": artifacts_ok,
        "sample_orders": order_dirs,
        "order_count": len(list(artifacts_dir.iterdir())) if artifacts_ok else 0,
    }
    if not artifacts_ok:
        result["status"] = "degraded"

    # Output directory
    output_dir = Path(settings.coscientist.output_dir)
    result["checks"]["output_dir"] = {
        "path": str(output_dir),
        "accessible": output_dir.exists(),
    }

    return result


@app.post("/run")
def run_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    """Start a new Co-Scientist session."""
    import uuid
    session_id = str(uuid.uuid4())[:8]

    _sessions[session_id] = {
        "status": "running",
        "state": None,
        "request": req.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
        "order_codes": req.resolved_order_codes,
    }
    _save_sessions()

    background_tasks.add_task(_execute_pipeline, session_id, req)

    return {"session_id": session_id, "status": "started"}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> SessionResponse:
    """Get current state of a Co-Scientist session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    state: CoScientistState | None = session.get("state")

    if state is None:
        # Fall back to persisted hypothesis data (survives restarts)
        return SessionResponse(
            session_id=session_id,
            status=session["status"],
            iteration=session.get("_iteration", 0),
            total_hypotheses=len(session.get("_hypotheses", [])),
            top_hypotheses=session.get("_hypotheses", [])[:10],
            experiment_designs=session.get("_experiment_designs", []),
            error=session.get("error"),
        )

    top_hyps = [h.to_dict() for h in state.hypotheses[:10]]
    exp_designs = [e.to_dict() for e in state.experiment_designs[:10]]

    return SessionResponse(
        session_id=session_id,
        status=session["status"],
        iteration=state.iteration,
        total_hypotheses=len(state.hypotheses),
        top_hypotheses=top_hyps,
        experiment_designs=exp_designs,
    )


@app.get("/session/{session_id}/discussion-packet")
def get_discussion_packet(session_id: str, max_hypotheses: int = 3) -> dict[str, Any]:
    """Return an evidence-gated, report-safe Discussion Evidence Packet.

    This endpoint intentionally returns interpretive candidates, not final report
    prose. PTM-platform should cite and render the packet under its own report
    rules after verifying the listed identifiers against its literature store.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    state: CoScientistState | None = session.get("state")
    max_hypotheses = max(1, min(max_hypotheses, 5))

    if state is not None:
        packet = build_discussion_evidence_packet(
            state,
            session_id=session_id,
            source_orders=session.get("order_codes", []),
            created_at=session.get("created_at"),
            max_hypotheses=max_hypotheses,
        )
        session["_discussion_packet"] = packet
        _save_sessions()
        return packet

    packet = session.get("_discussion_packet")
    if packet:
        return packet
    raise HTTPException(status_code=409, detail="No completed Co-Scientist result is available")


@app.get("/sessions")
def list_sessions(order_code: str = "", limit: int = 50):
    """List all sessions, optionally filtered by order_code."""
    rows = []
    for sid, s in _sessions.items():
        codes: list[str] = s.get("order_codes") or []
        if order_code and order_code not in codes:
            continue
        state: CoScientistState | None = s.get("state")
        hyp_count = len(state.hypotheses) if state else len(s.get("_hypotheses", []))
        rows.append({
            "session_id": sid,
            "status": s.get("status", "unknown"),
            "created_at": s.get("created_at", ""),
            "order_codes": codes,
            "total_hypotheses": hyp_count,
            "iteration": (state.iteration if state else s.get("_iteration", 0)),
            "research_goal": (s.get("request") or {}).get("research_goal", ""),
            "ptm_type": (s.get("request") or {}).get("ptm_type", ""),
        })
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return {"sessions": rows[:limit]}


@app.post("/session/{session_id}/feedback")
def submit_feedback(session_id: str, req: FeedbackRequest):
    """Submit scientist feedback and re-run with updated guidance."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    state: CoScientistState | None = session.get("state")

    if state is None:
        raise HTTPException(status_code=400, detail="Pipeline hasn't completed yet")

    state.scientist_feedback.append({
        "type": req.feedback_type,
        "content": req.content,
    })

    return {
        "status": "feedback_received",
        "total_feedback": len(state.scientist_feedback),
        "message": "Re-run the pipeline to incorporate feedback",
    }


@app.post("/session/{session_id}/cancel")
def cancel_pipeline(session_id: str):
    """Request cancellation of a running Co-Scientist session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    if session["status"] != "running":
        return {"session_id": session_id, "status": session["status"], "message": "Not running"}
    # Set a cancel flag; _execute_pipeline checks it each iteration
    session["cancel_requested"] = True
    session["status"] = "cancelling"
    _save_sessions()
    return {"session_id": session_id, "status": "cancelling"}


@app.post("/session/{session_id}/rerun")
def rerun_pipeline(session_id: str, background_tasks: BackgroundTasks):
    """
    Re-run the pipeline for an existing session, incorporating any scientist feedback.

    Reuses the original RunRequest (order_code, ptm_type, rag_collections, etc.)
    and appends the accumulated scientist_feedback as guidance for the new run.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    # Restore original request
    req_data = session.get("request", {})
    req = RunRequest(**req_data)

    # Mark as running again (preserving accumulated feedback on existing state)
    session["status"] = "running"

    background_tasks.add_task(_execute_pipeline, session_id, req)
    return {"session_id": session_id, "status": "restarted"}


@app.post("/session/{session_id}/design-experiments")
def design_experiments(session_id: str, top_n: int = 5):
    """Design experiments for the top hypotheses in a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    state: CoScientistState | None = session.get("state")

    if state is None or not state.hypotheses:
        raise HTTPException(status_code=400, detail="No hypotheses available")

    settings = get_settings()
    req_data = session.get("request", {})
    llm = _create_llm(
        settings,
        llm_provider=req_data.get("llm_provider", ""),
        llm_model=req_data.get("llm_model", ""),
    )

    designs = run_experiment_design(
        hypotheses=state.hypotheses,
        llm=llm,
        experimental_context=state.experimental_context,
        top_n=top_n,
    )

    state.experiment_designs = designs
    return {"designs": [d.to_dict() for d in designs]}


# ─── Background task ─────────────────────────────────────────────────────

def _execute_pipeline(session_id: str, req: RunRequest):
    """Execute the pipeline in background."""
    try:
        settings = get_settings()
        llm = _create_llm(settings, llm_provider=req.llm_provider, llm_model=req.llm_model)
        chromadb = ChromaDBConnector(settings.ptm_platform.chromadb_url)
        ptm_conn = PTMPlatformConnector(
            artifacts_dir=settings.ptm_platform.artifacts_dir,
            database_url=settings.ptm_platform.database_url,
        )

        pipeline = CoScientistPipeline(
            llm=llm,
            chromadb=chromadb,
            ptm_connector=ptm_conn,
            max_iterations=req.max_iterations,
            generate_candidates=settings.coscientist.generate_candidates,
            tournament_rounds=settings.coscientist.tournament_rounds,
            evolve_top_k=settings.coscientist.evolve_top_k,
            elo_k_factor=settings.coscientist.elo_k_factor,
        )

        # Get existing feedback if any
        existing_state = _sessions[session_id].get("state")
        feedback = existing_state.scientist_feedback if existing_state else []

        def _check_cancel() -> bool:
            return bool(_sessions.get(session_id, {}).get("cancel_requested"))

        state = pipeline.run(
            order_codes=req.resolved_order_codes,
            research_goal=req.research_goal,
            ptm_type=req.ptm_type,
            rag_collections=req.rag_collections,
            scientist_feedback=feedback,
            cancel_check=_check_cancel,
        )

        if _sessions[session_id].get("cancel_requested"):
            _sessions[session_id]["cancel_requested"] = False
            _sessions[session_id]["status"] = "cancelled"
            logger.info(f"Session {session_id} cancelled by user request")
            _sessions[session_id]["state"] = state  # preserve partial results
            _save_sessions()
            return

        # Auto-design experiments for top hypotheses
        designs = run_experiment_design(
            hypotheses=state.hypotheses,
            llm=llm,
            experimental_context=state.experimental_context,
            top_n=5,
        )
        state.experiment_designs = designs

        _sessions[session_id]["state"] = state
        _sessions[session_id]["status"] = "completed"
        discussion_packet = build_discussion_evidence_packet(
            state,
            session_id=session_id,
            source_orders=_sessions[session_id].get("order_codes", []),
            created_at=_sessions[session_id].get("created_at"),
        )
        _sessions[session_id]["_discussion_packet"] = discussion_packet
        _save_sessions()

        # Save full operational results plus the report-safe evidence packet.
        _save_results(session_id, state, settings.coscientist.output_dir, discussion_packet)

    except Exception as e:
        logger.exception("Pipeline failed for session %s", session_id)
        _sessions[session_id]["status"] = f"error: {e!s}"
        _save_sessions()


def _create_llm(settings, llm_provider: str = "", llm_model: str = "") -> LLMClient:
    """Create LLM client from settings, with optional per-request overrides.

    llm_provider: if non-empty, overrides settings.llm.provider
    llm_model:    if non-empty, used as the model name for the resolved provider
    """
    provider = llm_provider.strip() or settings.llm.provider

    # Derive per-provider model overrides from llm_model
    ollama_model  = llm_model if (llm_model and provider in ("ollama", "auto"))  else settings.llm.ollama_model
    openai_model  = llm_model if (llm_model and provider == "openai")  else settings.llm.openai_model
    gemini_model  = llm_model if (llm_model and provider == "gemini")  else settings.llm.gemini_model

    # When provider is auto and llm_model is given, set it for all backends
    if llm_model and provider == "auto":
        ollama_model = openai_model = gemini_model = llm_model

    return LLMClient(
        provider=provider,
        model=ollama_model,
        ollama_url=settings.llm.ollama_url,
        openai_api_key=settings.llm.openai_api_key,
        openai_model=openai_model,
        gemini_api_key=settings.llm.gemini_api_key,
        gemini_model=gemini_model,
    )


def _save_results(
    session_id: str,
    state: CoScientistState,
    output_dir: str,
    discussion_packet: dict[str, Any] | None = None,
):
    """Save operational results and an optional report-safe evidence packet to JSON."""
    out_path = Path(output_dir) / session_id
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "session_id": session_id,
        "research_goal": state.research_goal,
        "iteration": state.iteration,
        "tournament_history": state.tournament_history,
        "hypotheses": [h.to_dict() for h in state.hypotheses],
        "experiment_designs": [e.to_dict() for e in state.experiment_designs],
        "scientist_feedback": state.scientist_feedback,
        "discussion_evidence_packet": discussion_packet,
    }

    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path / 'results.json'}")

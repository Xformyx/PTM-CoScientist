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
from src.agents.meta_reviewer import run_meta_review
from src.agents.proximity import cluster_and_select_diverse_hypotheses
from src.connectors.chromadb_connector import ChromaDBConnector
from src.connectors.ptm_platform_connector import PTMPlatformConnector
from src.core.discussion_packet import build_discussion_evidence_packet
from src.core.evidence_graph import build_evidence_graph
from src.core.llm_client import LLMClient
from src.core.models import CoScientistState, LabResult
from src.core.pipeline import CoScientistPipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="PTM-CoScientist", version="0.1.0")

# ─── Session store — persisted to disk so restarts don't lose results ────
_SESSION_FILE = Path(
    os.getenv("COSCIENTIST_SESSION_FILE", "/data/coscientist/outputs/.sessions.json")
)
_sessions_lock = threading.RLock()


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
    """Persist sessions (excluding live state objects) to disk.

    Compacted reasoning fields are enough to restore a live ``CoScientistState``
    after process restart when ``results.json`` is unavailable.
    """
    p = _sessions_path()
    try:
        with _sessions_lock:
            serialisable = {}
            for sid, s in _sessions.items():
                entry = {k: v for k, v in s.items() if k != "state"}
                state: CoScientistState | None = s.get("state")
                if state is not None:
                    entry.update(_compact_state_fields(state, sid, s))
                serialisable[sid] = entry
        p.write_text(json.dumps(serialisable, default=str))
    except Exception as e:  # noqa: BLE001 - persistence failures are logged without breaking active sessions
        logger.warning(f"[Sessions] Failed to save sessions: {e}")


def _require_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return _sessions[session_id]


def _compact_state_fields(state: CoScientistState, session_id: str, session: dict[str, Any]) -> dict[str, Any]:
    """Serialize restore-critical fields used when the process restarts."""
    return {
        "_hypotheses": [hypothesis.to_dict() for hypothesis in state.hypotheses],
        "_iteration": state.iteration,
        "_max_iterations": state.max_iterations,
        "_experiment_designs": [design.to_dict() for design in state.experiment_designs],
        "_lab_results": [result.to_dict() for result in state.lab_results],
        "_scientist_feedback": state.scientist_feedback,
        "_experimental_context": state.experimental_context,
        "_enriched_ptm_data": state.enriched_ptm_data,
        "_rag_collections": state.rag_collections,
        "_tournament_history": state.tournament_history,
        "_evidence_graph": state.evidence_graph,
        "_evidence_graph_summary": state.evidence_graph.get("summary", {}),
        "_diversity_summary": state.diversity_summary,
        "_meta_review": state.meta_review,
        "_research_goal": state.research_goal,
        "_discussion_packet": build_discussion_evidence_packet(
            state,
            session_id=session_id,
            source_orders=session.get("order_codes", []),
            created_at=session.get("created_at"),
        ),
    }


def _results_path(session_id: str) -> Path:
    settings = get_settings()
    return Path(settings.coscientist.output_dir) / session_id / "results.json"


def _restore_state_from_results(session_id: str) -> CoScientistState | None:
    """Deserialize a completed session from ``results.json`` when present."""
    path = _results_path(session_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - corrupt result files should not take down the API
        logger.warning("[Sessions] Failed to read results for %s: %s", session_id, exc)
        return None
    if not isinstance(payload, dict):
        return None
    state = CoScientistState.from_dict(payload)
    if not state.hypotheses and not state.lab_results:
        return None
    if not state.evidence_graph and state.experimental_context:
        state.evidence_graph = build_evidence_graph(
            state.experimental_context,
            state.hypotheses,
            state.lab_results,
        )
    return state


def _restore_state_from_session_entry(session: dict[str, Any]) -> CoScientistState | None:
    """Rebuild live state from compacted session metadata."""
    hypotheses = session.get("_hypotheses") or []
    if not hypotheses and not session.get("_lab_results"):
        return None
    request = session.get("request") or {}
    context = dict(session.get("_experimental_context") or {})
    if "ptm_type" not in context and request.get("ptm_type"):
        context["ptm_type"] = request.get("ptm_type")
    state = CoScientistState.from_dict({
        "research_goal": session.get("_research_goal") or request.get("research_goal", ""),
        "experimental_context": context,
        "enriched_ptm_data": session.get("_enriched_ptm_data") or context.get("top_ptms") or [],
        "rag_collections": session.get("_rag_collections") or request.get("rag_collections") or [],
        "hypotheses": hypotheses,
        "experiment_designs": session.get("_experiment_designs") or [],
        "lab_results": session.get("_lab_results") or [],
        "scientist_feedback": session.get("_scientist_feedback") or [],
        "tournament_history": session.get("_tournament_history") or [],
        "evidence_graph": session.get("_evidence_graph") or {},
        "diversity_summary": session.get("_diversity_summary") or {},
        "meta_review": session.get("_meta_review") or {},
        "iteration": session.get("_iteration", 0),
        "max_iterations": session.get("_max_iterations") or request.get("max_iterations", 3),
    })
    if not state.evidence_graph and state.experimental_context:
        state.evidence_graph = build_evidence_graph(
            state.experimental_context,
            state.hypotheses,
            state.lab_results,
        )
    return state


def _ensure_live_state(session_id: str, *, require_hypotheses: bool = False) -> CoScientistState:
    """Return an in-memory state, restoring from disk after server restart if needed."""
    session = _require_session(session_id)
    with _sessions_lock:
        state: CoScientistState | None = session.get("state")
    if state is None:
        state = _restore_state_from_results(session_id) or _restore_state_from_session_entry(session)
        if state is None:
            raise HTTPException(
                status_code=409,
                detail="No restorable Co-Scientist reasoning state is available",
            )
        with _sessions_lock:
            session["state"] = state
        logger.info("[Sessions] Restored live state for session %s", session_id)
    if require_hypotheses and not state.hypotheses:
        raise HTTPException(status_code=400, detail="No hypotheses available")
    return state


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


class LabResultRequest(BaseModel):
    """Researcher-entered laboratory outcome for a specific hypothesis."""

    hypothesis_id: str
    outcome: str = "inconclusive"  # supports | contradicts | inconclusive
    assay_type: str = ""
    result_summary: str = ""
    observed_effect: str = ""
    controls: list[str] = []
    source_reference: str = ""


class SessionResponse(BaseModel):
    session_id: str
    status: str
    iteration: int
    total_hypotheses: int
    top_hypotheses: list
    experiment_designs: list
    error: str | None = None
    order_codes: list[str] = []


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
    if not req.resolved_order_codes:
        raise HTTPException(status_code=422, detail="order_codes or order_code is required")
    session_id = str(uuid.uuid4())[:8]

    with _sessions_lock:
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
    session = _require_session(session_id)
    with _sessions_lock:
        state: CoScientistState | None = session.get("state")
        order_codes = list(session.get("order_codes") or [])
        status = session["status"]
        error = session.get("error")
        compacted_hyps = session.get("_hypotheses", [])
        compacted_designs = session.get("_experiment_designs", [])
        compacted_iter = session.get("_iteration", 0)

    if state is None:
        # Fall back to persisted hypothesis data (survives restarts)
        return SessionResponse(
            session_id=session_id,
            status=status,
            iteration=compacted_iter,
            total_hypotheses=len(compacted_hyps),
            top_hypotheses=compacted_hyps[:10],
            experiment_designs=compacted_designs,
            error=error,
            order_codes=order_codes,
        )

    top_hyps = [h.to_dict() for h in state.hypotheses[:10]]
    exp_designs = [e.to_dict() for e in state.experiment_designs[:10]]

    return SessionResponse(
        session_id=session_id,
        status=status,
        iteration=state.iteration,
        total_hypotheses=len(state.hypotheses),
        top_hypotheses=top_hyps,
        experiment_designs=exp_designs,
        error=error,
        order_codes=order_codes,
    )


@app.get("/session/{session_id}/discussion-packet")
def get_discussion_packet(session_id: str, max_hypotheses: int = 3) -> dict[str, Any]:
    """Return an evidence-gated, report-safe Discussion Evidence Packet.

    This endpoint intentionally returns interpretive candidates, not final report
    prose. PTM-platform should cite and render the packet under its own report
    rules after verifying the listed identifiers against its literature store.
    """
    session = _require_session(session_id)
    with _sessions_lock:
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
    with _sessions_lock:
        items = list(_sessions.items())
    for sid, s in items:
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
    state = _ensure_live_state(session_id, require_hypotheses=True)
    state.scientist_feedback.append({
        "type": req.feedback_type,
        "content": req.content,
    })
    _save_sessions()

    return {
        "status": "feedback_received",
        "total_feedback": len(state.scientist_feedback),
        "message": "Re-run the pipeline to incorporate feedback",
    }


@app.post("/session/{session_id}/lab-results")
def submit_lab_result(session_id: str, req: LabResultRequest):
    """Record a researcher-observed lab outcome without overwriting source analysis.

    The outcome becomes explicit evidence for the next Reflection → Debate run.
    It does not automatically prove or reject a mechanism. After a server restart,
    state is restored from ``results.json`` or compacted session metadata.
    """
    state = _ensure_live_state(session_id, require_hypotheses=True)
    valid_ids = {hypothesis.id for hypothesis in state.hypotheses}
    if req.hypothesis_id not in valid_ids:
        raise HTTPException(status_code=404, detail="Hypothesis not found in this session")
    if req.outcome not in {"supports", "contradicts", "inconclusive"}:
        raise HTTPException(status_code=422, detail="outcome must be supports, contradicts, or inconclusive")

    result = LabResult(
        hypothesis_id=req.hypothesis_id,
        outcome=req.outcome,
        assay_type=req.assay_type,
        result_summary=req.result_summary,
        observed_effect=req.observed_effect,
        controls=req.controls,
        source_reference=req.source_reference,
    )
    state.lab_results.append(result)
    state.evidence_graph = build_evidence_graph(
        state.experimental_context,
        state.hypotheses,
        state.lab_results,
    )
    state.meta_review = {}
    settings = get_settings()
    discussion_packet = build_discussion_evidence_packet(
        state,
        session_id=session_id,
        source_orders=_sessions[session_id].get("order_codes", []),
        created_at=_sessions[session_id].get("created_at"),
    )
    _sessions[session_id]["_discussion_packet"] = discussion_packet
    _save_results(session_id, state, settings.coscientist.output_dir, discussion_packet)
    _save_sessions()
    return {
        "status": "lab_result_recorded",
        "lab_result": result.to_dict(),
        "message": "Re-run the pipeline to incorporate this evidence into reflection, debate, and ranking.",
    }


@app.get("/session/{session_id}/scientific-reasoning")
def get_scientific_reasoning(session_id: str) -> dict[str, Any]:
    """Return graph, diversity, reflection, meta-review, and lab-result provenance."""
    state = _ensure_live_state(session_id)
    return {
        "session_id": session_id,
        "evidence_graph": state.evidence_graph,
        "diversity_summary": state.diversity_summary,
        "meta_review": state.meta_review,
        "hypothesis_reflections": [
            {"hypothesis_id": hypothesis.id, "reflection": hypothesis.reflection}
            for hypothesis in state.hypotheses
        ],
        "lab_results": [result.to_dict() for result in state.lab_results],
    }


@app.post("/session/{session_id}/cancel")
def cancel_pipeline(session_id: str):
    """Request cancellation of a running Co-Scientist session."""
    session = _require_session(session_id)
    with _sessions_lock:
        if session["status"] != "running":
            return {"session_id": session_id, "status": session["status"], "message": "Not running"}
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
    session = _require_session(session_id)
    with _sessions_lock:
        if session["status"] in {"running", "cancelling"}:
            raise HTTPException(status_code=409, detail="Pipeline is already running or cancelling")
        req_data = session.get("request", {})
        session["status"] = "running"
        session["cancel_requested"] = False
        session["error"] = None
        _save_sessions()
    req = RunRequest(**req_data)

    background_tasks.add_task(_execute_pipeline, session_id, req)
    return {"session_id": session_id, "status": "restarted"}


@app.post("/session/{session_id}/design-experiments")
def design_experiments(session_id: str, top_n: int = 5):
    """Design experiments for the top hypotheses in a session."""
    state = _ensure_live_state(session_id, require_hypotheses=True)
    session = _sessions[session_id]
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
    _save_sessions()
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
            reflection_enabled=settings.coscientist.reflection_enabled,
            evidence_graph_enabled=settings.coscientist.evidence_graph_enabled,
            proximity_enabled=settings.coscientist.proximity_enabled,
            max_diverse_hypotheses=settings.coscientist.max_diverse_hypotheses,
            max_hypotheses=settings.coscientist.max_hypotheses,
        )

        # Restore prior feedback/lab results/hypotheses after restart when needed.
        # Fresh /run sessions have no restorable state yet; that is expected.
        with _sessions_lock:
            existing_state: CoScientistState | None = _sessions[session_id].get("state")
            session_snapshot = _sessions[session_id]
        if existing_state is None:
            existing_state = (
                _restore_state_from_results(session_id)
                or _restore_state_from_session_entry(session_snapshot)
            )
            if existing_state is not None:
                with _sessions_lock:
                    _sessions[session_id]["state"] = existing_state
        feedback = existing_state.scientist_feedback if existing_state else []
        lab_results = existing_state.lab_results if existing_state else []
        prior_hypotheses = existing_state.hypotheses if existing_state else []

        def _check_cancel() -> bool:
            with _sessions_lock:
                return bool(_sessions.get(session_id, {}).get("cancel_requested"))

        state = pipeline.run(
            order_codes=req.resolved_order_codes,
            research_goal=req.research_goal,
            ptm_type=req.ptm_type,
            rag_collections=req.rag_collections,
            scientist_feedback=feedback,
            lab_results=lab_results,
            prior_hypotheses=prior_hypotheses,
            cancel_check=_check_cancel,
        )

        # Persist core results immediately so a later design/meta-review failure
        # cannot discard a finished Generate → Debate → Evolve run.
        with _sessions_lock:
            _sessions[session_id]["state"] = state
            cancelled = bool(_sessions[session_id].get("cancel_requested"))
            if cancelled:
                _sessions[session_id]["cancel_requested"] = False
                _sessions[session_id]["status"] = "cancelled"
            _save_sessions()

        if cancelled:
            logger.info(f"Session {session_id} cancelled by user request")
            _save_results(session_id, state, settings.coscientist.output_dir)
            return

        post_error = None
        try:
            if settings.coscientist.proximity_enabled:
                selected_hypotheses, state.diversity_summary = cluster_and_select_diverse_hypotheses(
                    state.hypotheses,
                    max_hypotheses=settings.coscientist.max_diverse_hypotheses,
                )
            else:
                selected_hypotheses = state.hypotheses[:settings.coscientist.max_diverse_hypotheses]

            designs = run_experiment_design(
                hypotheses=selected_hypotheses,
                llm=llm,
                experimental_context=state.experimental_context,
                top_n=len(selected_hypotheses),
            )
            state.experiment_designs = designs
            state.evidence_graph = build_evidence_graph(
                state.experimental_context,
                state.hypotheses,
                state.lab_results,
            )
            if settings.coscientist.meta_review_enabled:
                state.meta_review = run_meta_review(
                    research_goal=state.research_goal,
                    hypotheses=selected_hypotheses,
                    evidence_graph_summary=state.evidence_graph.get("summary", {}),
                    experiment_designs=state.experiment_designs,
                    lab_results=state.lab_results,
                    scientist_feedback=state.scientist_feedback,
                    llm=llm,
                )
        except Exception as exc:  # noqa: BLE001 - keep hypothesis results even if enrichment fails
            logger.exception("Post-pipeline enrichment failed for session %s", session_id)
            post_error = str(exc)

        discussion_packet = build_discussion_evidence_packet(
            state,
            session_id=session_id,
            source_orders=_sessions[session_id].get("order_codes", []),
            created_at=_sessions[session_id].get("created_at"),
        )
        with _sessions_lock:
            _sessions[session_id]["state"] = state
            _sessions[session_id]["status"] = "completed"
            _sessions[session_id]["error"] = f"Completed with warnings: {post_error}" if post_error else None
            _sessions[session_id]["_discussion_packet"] = discussion_packet
            _save_sessions()

        _save_results(session_id, state, settings.coscientist.output_dir, discussion_packet)

    except Exception as e:
        logger.exception("Pipeline failed for session %s", session_id)
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["status"] = f"error: {e!s}"
                _sessions[session_id]["error"] = str(e)
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

    results = state.to_dict()
    results["session_id"] = session_id
    results["discussion_evidence_packet"] = discussion_packet

    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path / 'results.json'}")

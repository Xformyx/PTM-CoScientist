"""Core data models for PTM-CoScientist scientific reasoning."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    GENERATED = "generated"
    REFLECTED = "reflected"
    DEBATED = "debated"
    EVOLVED = "evolved"
    VALIDATED = "validated"
    REJECTED = "rejected"


class HypothesisCategory(str, Enum):
    MECHANISTIC = "mechanistic"
    TEMPORAL = "temporal"
    PREDICTIVE = "predictive"
    INTEGRATIVE = "integrative"
    THERAPEUTIC = "therapeutic"


@dataclass
class Hypothesis:
    """A falsifiable IF–THEN–BECAUSE hypothesis with provenance."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition: str = ""  # IF
    prediction: str = ""  # THEN
    mechanism: str = ""  # BECAUSE
    category: HypothesisCategory = HypothesisCategory.MECHANISTIC
    supporting_ptms: list[str] = field(default_factory=list)
    signaling_chain: str = ""
    testable_prediction: str = ""
    confidence: float = 0.5
    elo_rating: int = 1500
    status: HypothesisStatus = HypothesisStatus.GENERATED

    # Literature and debate provenance
    evidence_for: list[dict[str, Any]] = field(default_factory=list)
    evidence_against: list[dict[str, Any]] = field(default_factory=list)
    debate_history: list[dict[str, Any]] = field(default_factory=list)

    # Reflection is deliberately separate from peer debate. It records atomic
    # claims, evidence gaps, confounders, falsification conditions and a review
    # recommendation before the hypothesis enters the tournament.
    reflection: dict[str, Any] = field(default_factory=dict)

    # Proximity information prevents near-duplicate hypotheses from dominating
    # the final shortlist.
    proximity_cluster: str = ""

    # Evolution lineage enables an audit trail from an evolved claim to its
    # debated parents and the critique(s) it addressed.
    parent_hypothesis_ids: list[str] = field(default_factory=list)
    evolution_type: str = ""  # strengthened | combined | deepened | divergent
    addressed_critiques: list[str] = field(default_factory=list)

    generation_round: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "prediction": self.prediction,
            "mechanism": self.mechanism,
            "category": self.category.value,
            "supporting_ptms": self.supporting_ptms,
            "signaling_chain": self.signaling_chain,
            "testable_prediction": self.testable_prediction,
            "confidence": self.confidence,
            "elo_rating": self.elo_rating,
            "status": self.status.value,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "debate_history": self.debate_history,
            "reflection": self.reflection,
            "proximity_cluster": self.proximity_cluster,
            "parent_hypothesis_ids": self.parent_hypothesis_ids,
            "evolution_type": self.evolution_type,
            "addressed_critiques": self.addressed_critiques,
            "generation_round": self.generation_round,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        """Rehydrate a hypothesis from persisted JSON without inventing new fields."""
        payload = data or {}
        category_raw = str(payload.get("category") or HypothesisCategory.MECHANISTIC.value)
        status_raw = str(payload.get("status") or HypothesisStatus.GENERATED.value)
        try:
            category = HypothesisCategory(category_raw)
        except ValueError:
            category = HypothesisCategory.MECHANISTIC
        try:
            status = HypothesisStatus(status_raw)
        except ValueError:
            status = HypothesisStatus.GENERATED
        return cls(
            id=str(payload["id"]) if payload.get("id") else str(uuid.uuid4())[:8],
            condition=str(payload.get("condition") or ""),
            prediction=str(payload.get("prediction") or ""),
            mechanism=str(payload.get("mechanism") or ""),
            category=category,
            supporting_ptms=list(payload.get("supporting_ptms") or []),
            signaling_chain=str(payload.get("signaling_chain") or ""),
            testable_prediction=str(payload.get("testable_prediction") or ""),
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            elo_rating=int(payload.get("elo_rating", 1500) or 1500),
            status=status,
            evidence_for=list(payload.get("evidence_for") or []),
            evidence_against=list(payload.get("evidence_against") or []),
            debate_history=list(payload.get("debate_history") or []),
            reflection=dict(payload.get("reflection") or {}),
            proximity_cluster=str(payload.get("proximity_cluster") or ""),
            parent_hypothesis_ids=list(payload.get("parent_hypothesis_ids") or []),
            evolution_type=str(payload.get("evolution_type") or ""),
            addressed_critiques=list(payload.get("addressed_critiques") or []),
            generation_round=int(payload.get("generation_round", 0) or 0),
            created_at=float(payload.get("created_at") or time.time()),
        )


@dataclass
class ExperimentDesign:
    """A structured proposal to falsify or support a hypothesis."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    title: str = ""
    objective: str = ""
    approach: str = ""
    key_reagents: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    alternative_outcome: str = ""
    estimated_timeline: str = ""
    priority: str = "medium"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "objective": self.objective,
            "approach": self.approach,
            "key_reagents": self.key_reagents,
            "controls": self.controls,
            "expected_outcome": self.expected_outcome,
            "alternative_outcome": self.alternative_outcome,
            "estimated_timeline": self.estimated_timeline,
            "priority": self.priority,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentDesign":
        payload = data or {}
        return cls(
            id=str(payload["id"]) if payload.get("id") else str(uuid.uuid4())[:8],
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            title=str(payload.get("title") or ""),
            objective=str(payload.get("objective") or ""),
            approach=str(payload.get("approach") or ""),
            key_reagents=list(payload.get("key_reagents") or []),
            controls=list(payload.get("controls") or []),
            expected_outcome=str(payload.get("expected_outcome") or ""),
            alternative_outcome=str(payload.get("alternative_outcome") or ""),
            estimated_timeline=str(payload.get("estimated_timeline") or ""),
            priority=str(payload.get("priority") or "medium"),
            rationale=str(payload.get("rationale") or ""),
        )


@dataclass
class LabResult:
    """Researcher-entered experimental evidence for a candidate hypothesis.

    This object stores an observed outcome; it does not automatically establish
    causality or overwrite the underlying PTM-platform analysis.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    outcome: str = "inconclusive"  # supports | contradicts | inconclusive
    assay_type: str = ""
    result_summary: str = ""
    observed_effect: str = ""
    controls: list[str] = field(default_factory=list)
    source_reference: str = ""
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "outcome": self.outcome,
            "assay_type": self.assay_type,
            "result_summary": self.result_summary,
            "observed_effect": self.observed_effect,
            "controls": self.controls,
            "source_reference": self.source_reference,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabResult":
        payload = data or {}
        outcome = str(payload.get("outcome") or "inconclusive")
        if outcome not in {"supports", "contradicts", "inconclusive"}:
            outcome = "inconclusive"
        return cls(
            id=str(payload["id"]) if payload.get("id") else str(uuid.uuid4())[:8],
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            outcome=outcome,
            assay_type=str(payload.get("assay_type") or ""),
            result_summary=str(payload.get("result_summary") or ""),
            observed_effect=str(payload.get("observed_effect") or ""),
            controls=list(payload.get("controls") or []),
            source_reference=str(payload.get("source_reference") or ""),
            recorded_at=float(payload.get("recorded_at") or time.time()),
        )


@dataclass
class CoScientistState:
    """Shared state flowing through the Co-Scientist pipeline."""

    # Input context from PTM-platform; this is read-only source material.
    order_id: int | None = None
    research_goal: str = ""
    experimental_context: dict[str, Any] = field(default_factory=dict)
    enriched_ptm_data: list[dict[str, Any]] = field(default_factory=list)
    kinase_modules: dict[str, Any] = field(default_factory=dict)
    signal_flow: dict[str, Any] = field(default_factory=dict)
    comovement_clusters: dict[str, Any] = field(default_factory=dict)
    rag_collections: list[str] = field(default_factory=list)

    # Scientific reasoning state
    hypotheses: list[Hypothesis] = field(default_factory=list)
    tournament_history: list[dict[str, Any]] = field(default_factory=list)
    experiment_designs: list[ExperimentDesign] = field(default_factory=list)
    evidence_graph: dict[str, Any] = field(default_factory=dict)
    diversity_summary: dict[str, Any] = field(default_factory=dict)
    meta_review: dict[str, Any] = field(default_factory=dict)
    lab_results: list[LabResult] = field(default_factory=list)
    final_report: str = ""

    # Scientist-in-the-loop
    scientist_feedback: list[dict[str, str]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full operational state for results.json / session restore."""
        return {
            "order_id": self.order_id,
            "research_goal": self.research_goal,
            "experimental_context": self.experimental_context,
            "enriched_ptm_data": self.enriched_ptm_data,
            "kinase_modules": self.kinase_modules,
            "signal_flow": self.signal_flow,
            "comovement_clusters": self.comovement_clusters,
            "rag_collections": self.rag_collections,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "tournament_history": self.tournament_history,
            "experiment_designs": [design.to_dict() for design in self.experiment_designs],
            "evidence_graph": self.evidence_graph,
            "diversity_summary": self.diversity_summary,
            "meta_review": self.meta_review,
            "lab_results": [result.to_dict() for result in self.lab_results],
            "scientist_feedback": self.scientist_feedback,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoScientistState":
        """Restore a live state object from persisted results or session metadata."""
        payload = data or {}
        context = dict(payload.get("experimental_context") or {})
        enriched = list(payload.get("enriched_ptm_data") or context.get("top_ptms") or [])
        return cls(
            order_id=payload.get("order_id"),
            research_goal=str(payload.get("research_goal") or ""),
            experimental_context=context,
            enriched_ptm_data=enriched,
            kinase_modules=dict(payload.get("kinase_modules") or context.get("kinase_modules") or {}),
            signal_flow=dict(payload.get("signal_flow") or context.get("signal_flow") or {}),
            comovement_clusters=dict(
                payload.get("comovement_clusters") or context.get("comovement_clusters") or {}
            ),
            rag_collections=list(payload.get("rag_collections") or []),
            hypotheses=[
                Hypothesis.from_dict(item)
                for item in (payload.get("hypotheses") or [])
                if isinstance(item, dict)
            ],
            tournament_history=list(payload.get("tournament_history") or []),
            experiment_designs=[
                ExperimentDesign.from_dict(item)
                for item in (payload.get("experiment_designs") or [])
                if isinstance(item, dict)
            ],
            evidence_graph=dict(payload.get("evidence_graph") or {}),
            diversity_summary=dict(payload.get("diversity_summary") or {}),
            meta_review=dict(payload.get("meta_review") or {}),
            lab_results=[
                LabResult.from_dict(item)
                for item in (payload.get("lab_results") or [])
                if isinstance(item, dict)
            ],
            scientist_feedback=[
                item for item in (payload.get("scientist_feedback") or []) if isinstance(item, dict)
            ],
            iteration=int(payload.get("iteration", 0) or 0),
            max_iterations=int(payload.get("max_iterations", 3) or 3),
        )

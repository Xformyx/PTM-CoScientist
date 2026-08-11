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

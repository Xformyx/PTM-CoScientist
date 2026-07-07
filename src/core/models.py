"""
Core data models for PTM-CoScientist.

Defines the structured types flowing through the agent pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import time


class HypothesisStatus(str, Enum):
    GENERATED = "generated"
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
    """A structured scientific hypothesis with IF-THEN-BECAUSE format."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition: str = ""  # IF
    prediction: str = ""  # THEN
    mechanism: str = ""  # BECAUSE
    category: HypothesisCategory = HypothesisCategory.MECHANISTIC
    supporting_ptms: List[str] = field(default_factory=list)
    signaling_chain: str = ""  # e.g., "EGFR → SRC → VIM-S56"
    testable_prediction: str = ""
    confidence: float = 0.5
    elo_rating: int = 1500
    status: HypothesisStatus = HypothesisStatus.GENERATED
    evidence_for: List[Dict[str, Any]] = field(default_factory=list)
    evidence_against: List[Dict[str, Any]] = field(default_factory=list)
    debate_history: List[Dict[str, str]] = field(default_factory=list)
    generation_round: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
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
            "generation_round": self.generation_round,
        }


@dataclass
class ExperimentDesign:
    """A structured experiment design proposal."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    title: str = ""
    objective: str = ""
    approach: str = ""  # e.g., "Western Blot", "LC-MS/MS", "Kinase Inhibitor Assay"
    key_reagents: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    alternative_outcome: str = ""
    estimated_timeline: str = ""
    priority: str = "medium"  # high | medium | low
    rationale: str = ""

    def to_dict(self) -> dict:
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
class CoScientistState:
    """Shared state flowing through the Co-Scientist pipeline."""

    # Input context (from PTM-platform)
    order_id: Optional[int] = None
    research_goal: str = ""
    experimental_context: Dict[str, Any] = field(default_factory=dict)
    enriched_ptm_data: List[Dict[str, Any]] = field(default_factory=list)
    kinase_modules: Dict[str, Any] = field(default_factory=dict)
    signal_flow: Dict[str, Any] = field(default_factory=dict)
    comovement_clusters: Dict[str, Any] = field(default_factory=dict)
    rag_collections: List[str] = field(default_factory=list)

    # Pipeline state
    hypotheses: List[Hypothesis] = field(default_factory=list)
    tournament_history: List[Dict[str, Any]] = field(default_factory=list)
    experiment_designs: List[ExperimentDesign] = field(default_factory=list)
    final_report: str = ""

    # Scientist-in-the-loop
    scientist_feedback: List[Dict[str, str]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3

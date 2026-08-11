"""Meta-review Agent — synthesise a bounded scientific reasoning summary."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.llm_client import LLMClient
from src.core.models import ExperimentDesign, Hypothesis, LabResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior PTM research meta-reviewer. Synthesize the supplied candidate
hypotheses, their reflection concerns, literature support/contradiction,
experiment proposals, and laboratory outcomes. You must separate observed facts
from candidate interpretations. Do not invent data, citations, or experimental
results. The result is a decision-support summary, not a final scientific claim.

Return ONLY a JSON object:
{
  "executive_summary": "...",
  "leading_mechanism": {"hypothesis_id": "...", "rationale": "..."},
  "alternative_models": [{"hypothesis_id": "...", "rationale": "..."}],
  "key_uncertainties": ["..."],
  "next_best_experiment": {"hypothesis_id": "...", "rationale": "...", "discriminates_between": ["..."]},
  "researcher_questions": ["..."],
  "usage_notice": "Interpretive candidates require researcher review and experimental validation."
}
"""


def run_meta_review(
    *,
    research_goal: str,
    hypotheses: list[Hypothesis],
    evidence_graph_summary: dict[str, Any],
    experiment_designs: list[ExperimentDesign],
    lab_results: list[LabResult],
    scientist_feedback: list[dict[str, str]],
    llm: LLMClient,
) -> dict[str, Any]:
    """Generate a final audit-friendly synthesis of the selected candidates."""
    if not hypotheses:
        return _fallback("No eligible hypotheses were available for meta-review.")

    prompt = _build_prompt(
        research_goal,
        hypotheses,
        evidence_graph_summary,
        experiment_designs,
        lab_results,
        scientist_feedback,
    )
    response = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2, max_tokens=2200)
    review = _parse(response)
    review["reviewed_hypothesis_ids"] = [hypothesis.id for hypothesis in hypotheses]
    return review


def _build_prompt(
    research_goal: str,
    hypotheses: list[Hypothesis],
    graph_summary: dict[str, Any],
    designs: list[ExperimentDesign],
    lab_results: list[LabResult],
    feedback: list[dict[str, str]],
) -> str:
    candidates = []
    for hypothesis in hypotheses:
        candidates.append({
            "id": hypothesis.id,
            "claim": f"{hypothesis.condition} {hypothesis.prediction} {hypothesis.mechanism}",
            "supporting_ptms": hypothesis.supporting_ptms,
            "signaling_chain": hypothesis.signaling_chain,
            "reflection": hypothesis.reflection,
            "support_count": len(hypothesis.evidence_for),
            "contradiction_count": len(hypothesis.evidence_against),
            "testable_prediction": hypothesis.testable_prediction,
            "proximity_cluster": hypothesis.proximity_cluster,
        })
    relevant_designs = [design.to_dict() for design in designs if design.hypothesis_id in {h.id for h in hypotheses}]
    return "\n".join([
        "## Research Goal",
        research_goal or "No explicit research goal supplied.",
        "\n## Evidence Graph Summary",
        json.dumps(graph_summary, ensure_ascii=False),
        "\n## Diverse Candidate Hypotheses",
        json.dumps(candidates, ensure_ascii=False)[:9000],
        "\n## Proposed Experiments",
        json.dumps(relevant_designs, ensure_ascii=False)[:6000],
        "\n## Laboratory Results Entered by Researcher",
        json.dumps([result.to_dict() for result in lab_results], ensure_ascii=False)[:3500],
        "\n## Researcher Feedback",
        json.dumps(feedback, ensure_ascii=False)[:2500],
    ])


def _parse(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            payload = json.loads(text[start:end + 1])
            if isinstance(payload, dict):
                return _normalise(payload)
        except json.JSONDecodeError:
            pass
    logger.warning("[MetaReview] Could not parse response")
    return _fallback("Meta-review model response could not be parsed; manual synthesis required.")


def _normalise(payload: dict[str, Any]) -> dict[str, Any]:
    def as_list(key: str) -> list[Any]:
        value = payload.get(key, [])
        return value if isinstance(value, list) else []

    return {
        "executive_summary": str(payload.get("executive_summary") or ""),
        "leading_mechanism": payload.get("leading_mechanism") if isinstance(payload.get("leading_mechanism"), dict) else {},
        "alternative_models": [item for item in as_list("alternative_models") if isinstance(item, dict)],
        "key_uncertainties": [str(item) for item in as_list("key_uncertainties") if str(item).strip()],
        "next_best_experiment": payload.get("next_best_experiment") if isinstance(payload.get("next_best_experiment"), dict) else {},
        "researcher_questions": [str(item) for item in as_list("researcher_questions") if str(item).strip()],
        "usage_notice": str(payload.get("usage_notice") or "Interpretive candidates require researcher review and experimental validation."),
    }


def _fallback(summary: str) -> dict[str, Any]:
    return {
        "executive_summary": summary,
        "leading_mechanism": {},
        "alternative_models": [],
        "key_uncertainties": [],
        "next_best_experiment": {},
        "researcher_questions": [],
        "usage_notice": "Interpretive candidates require researcher review and experimental validation.",
    }

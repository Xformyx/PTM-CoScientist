"""
Experiment Designer Agent — Designs validation experiments for top hypotheses.

Generates concrete experimental protocols including:
- Approach selection (Western Blot, LC-MS/MS, Kinase Inhibitor Assay, etc.)
- Key reagents (antibodies, inhibitors, cell lines)
- Controls (positive, negative, vehicle)
- Expected vs. alternative outcomes
- Priority ranking based on feasibility and impact
"""

import json
import logging
from typing import List, Dict, Any

from src.core.llm_client import LLMClient
from src.core.models import Hypothesis, ExperimentDesign

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert in designing PTM validation experiments.
Given a hypothesis about post-translational modifications and cell signaling,
design a concrete, actionable experiment to test it.

## EXPERTISE AREAS
- Phosphoproteomics (LC-MS/MS, TiO2 enrichment, TMT labeling)
- Western Blot with phospho-specific antibodies
- Kinase inhibitor assays (small molecule, siRNA knockdown)
- Site-directed mutagenesis (phospho-dead/mimetic mutants)
- Proximity ligation assay (PLA) for protein interactions
- FRET/BRET for real-time kinase activity
- In vitro kinase assays

## RULES
1. Be SPECIFIC: name actual antibodies, inhibitors, cell lines when possible
2. Include proper controls (vehicle, kinase-dead mutant, scrambled siRNA)
3. Consider feasibility: prioritize experiments that can be done in 2-4 weeks
4. Propose both a primary experiment and a complementary validation
5. State what result would CONFIRM vs. REFUTE the hypothesis

## OUTPUT FORMAT
Return ONLY a valid JSON array:
[
  {
    "title": "Short descriptive title",
    "objective": "What this experiment tests",
    "approach": "Primary technique",
    "key_reagents": ["Reagent 1", "Reagent 2"],
    "controls": ["Control 1", "Control 2"],
    "expected_outcome": "Result if hypothesis is correct",
    "alternative_outcome": "Result if hypothesis is wrong",
    "estimated_timeline": "e.g., 2-3 weeks",
    "priority": "high | medium | low",
    "rationale": "Why this approach is optimal for this hypothesis"
  }
]
"""


def run_experiment_design(
    hypotheses: List[Hypothesis],
    llm: LLMClient,
    experimental_context: Dict[str, Any] = None,
    top_n: int = 5,
) -> List[ExperimentDesign]:
    """
    Design experiments for the top-ranked hypotheses.

    Args:
        hypotheses: Ranked hypotheses (top first)
        llm: LLM client
        experimental_context: Lab context (cell type, available equipment, etc.)
        top_n: Number of top hypotheses to design experiments for

    Returns:
        List of ExperimentDesign objects
    """
    if not hypotheses:
        return []

    top = hypotheses[:top_n]
    all_designs = []

    for h in top:
        designs = _design_for_hypothesis(h, llm, experimental_context)
        all_designs.extend(designs)

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_designs.sort(key=lambda d: priority_order.get(d.priority, 1))

    logger.info(f"[ExperimentDesigner] Designed {len(all_designs)} experiments for top {len(top)} hypotheses")
    return all_designs


def _design_for_hypothesis(
    hypothesis: Hypothesis,
    llm: LLMClient,
    experimental_context: Dict[str, Any] = None,
) -> List[ExperimentDesign]:
    """Design experiments for a single hypothesis."""
    prompt = _build_prompt(hypothesis, experimental_context)

    response = llm.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=3000,
    )

    designs = _parse_response(response, hypothesis.id)
    return designs


def _build_prompt(hypothesis: Hypothesis, context: Dict[str, Any] = None) -> str:
    """Build experiment design prompt."""
    parts = [
        "## Hypothesis to Test",
        f"- IF: {hypothesis.condition}",
        f"- THEN: {hypothesis.prediction}",
        f"- BECAUSE: {hypothesis.mechanism}",
        f"- Signaling chain: {hypothesis.signaling_chain}",
        f"- Supporting PTMs: {', '.join(hypothesis.supporting_ptms[:5])}",
        f"- Confidence: {hypothesis.confidence} (Elo: {hypothesis.elo_rating})",
        f"- Testable prediction: {hypothesis.testable_prediction}",
    ]

    # Evidence context
    if hypothesis.evidence_for:
        parts.append(f"\n## Supporting Literature ({len(hypothesis.evidence_for)} papers)")
        for ev in hypothesis.evidence_for[:3]:
            parts.append(f"- {ev.get('source', 'Unknown')}: {ev.get('text', '')[:150]}")

    # Lab context
    if context:
        ctx = context.get("experimental_context", context)
        if isinstance(ctx, dict):
            lab_info = []
            if ctx.get("cell_type"):
                lab_info.append(f"Cell type: {ctx['cell_type']}")
            if ctx.get("species"):
                lab_info.append(f"Species: {ctx['species']}")
            if ctx.get("treatment"):
                lab_info.append(f"Treatment: {ctx['treatment']}")
            if lab_info:
                parts.append("\n## Lab Context\n" + "\n".join(lab_info))

    parts.append("\n## Task\nDesign 1-2 experiments to validate this hypothesis.")
    parts.append("Include a primary experiment and optionally a complementary validation.")

    return "\n".join(parts)


def _parse_response(response: str, hypothesis_id: str) -> List[ExperimentDesign]:
    """Parse experiment designs from LLM response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[ExperimentDesigner] Could not find JSON array")
        return []

    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error(f"[ExperimentDesigner] JSON parse error: {e}")
        return []

    designs = []
    for item in items:
        if not isinstance(item, dict):
            continue

        d = ExperimentDesign(
            hypothesis_id=hypothesis_id,
            title=item.get("title", ""),
            objective=item.get("objective", ""),
            approach=item.get("approach", ""),
            key_reagents=item.get("key_reagents", []),
            controls=item.get("controls", []),
            expected_outcome=item.get("expected_outcome", ""),
            alternative_outcome=item.get("alternative_outcome", ""),
            estimated_timeline=item.get("estimated_timeline", ""),
            priority=item.get("priority", "medium"),
            rationale=item.get("rationale", ""),
        )

        if d.title and d.approach:
            designs.append(d)

    return designs

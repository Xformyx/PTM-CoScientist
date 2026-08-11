"""
Evolver Agent — Refines and combines top-ranked hypotheses.

Inspired by Google Co-Scientist's Evolution agent.
Takes the highest-Elo hypotheses, addresses their weaknesses
(from debate critiques), and produces improved versions with explicit lineage.
"""

import json
import logging
from typing import Any

from src.core.llm_client import LLMClient
from src.core.models import Hypothesis, HypothesisCategory, HypothesisStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a hypothesis evolution specialist in PTM biology.
You receive top-ranked hypotheses that have survived a scientific debate tournament,
along with critiques from the debate.

Your task: EVOLVE these hypotheses by:
1. **Strengthening** — Address the critiques to make hypotheses more robust
2. **Combining** — Merge complementary hypotheses into more powerful integrative ones
3. **Deepening** — Add mechanistic detail based on the signaling context
4. **Diverging** — Propose one unexpected angle that none of the inputs considered

## RULES
- Each evolved hypothesis must be MORE specific than its parent
- Include concrete PTM sites and signaling chains
- Address at least one critique from the debate unless evolution_type is divergent
- Maintain testability — every hypothesis needs a clear experimental prediction
- `parent_ids` MUST contain only IDs present in the input hypotheses

## OUTPUT FORMAT
Return ONLY a valid JSON array:
[
  {
    "condition": "IF ...",
    "prediction": "THEN ...",
    "mechanism": "BECAUSE ...",
    "category": "mechanistic | temporal | predictive | integrative | therapeutic",
    "supporting_ptms": ["GENE-S123"],
    "signaling_chain": "RECEPTOR → KINASE → SUBSTRATE",
    "testable_prediction": "...",
    "evolution_type": "strengthened | combined | deepened | divergent",
    "parent_ids": ["id1", "id2"],
    "addressed_critiques": ["Specific critique addressed by this evolution"]
  }
]
"""


def run_evolution(
    hypotheses: list[Hypothesis],
    llm: LLMClient,
    top_k: int = 3,
    context: dict[str, Any] | None = None,
) -> list[Hypothesis]:
    """Evolve top-ranked hypotheses into improved versions with traceable lineage."""
    if not hypotheses:
        return []

    top = hypotheses[:top_k]
    user_prompt = _build_evolution_prompt(top, context)

    response = llm.generate(
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=4096,
    )

    evolved = _parse_response(response, top)
    logger.info(f"[Evolver] Evolved {len(evolved)} hypotheses from top {top_k}")
    return evolved


def _build_evolution_prompt(top_hypotheses: list[Hypothesis], context: dict[str, Any] | None = None) -> str:
    """Build prompt for evolution."""
    parts = ["## Top-Ranked Hypotheses (from debate tournament)\n"]
    for i, h in enumerate(top_hypotheses):
        parts.append(f"### Hypothesis {i + 1} (Elo: {h.elo_rating}, ID: {h.id})")
        parts.append(f"- IF: {h.condition}")
        parts.append(f"- THEN: {h.prediction}")
        parts.append(f"- BECAUSE: {h.mechanism}")
        parts.append(f"- Signaling: {h.signaling_chain}")
        parts.append(f"- Supporting PTMs: {', '.join(h.supporting_ptms[:5])}")
        parts.append(f"- Confidence: {h.confidence}")
        parts.append(f"- Literature support: {len(h.evidence_for)} papers")

        critiques = [d.get("critique", "") for d in h.debate_history if d.get("critique")]
        if critiques:
            parts.append(f"- **Critiques to address:** {'; '.join(critiques[:3])}")
        parts.append("")

    if context:
        kinase = context.get("kinase_modules", {})
        if kinase:
            parts.append(f"## Available Kinase Context\n{json.dumps(kinase, indent=2)[:1000]}")

    parts.append(f"\n## Task\nEvolve these {len(top_hypotheses)} hypotheses into 3-4 improved versions.")
    parts.append("At least one should COMBINE insights from multiple parent hypotheses.")
    parts.append("At least one should be a DIVERGENT angle not covered by any parent.")

    return "\n".join(parts)


def _parse_response(response: str, parents: list[Hypothesis]) -> list[Hypothesis]:
    """Parse evolved hypotheses and retain model-declared, validated lineage."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[Evolver] Could not find JSON array in response")
        return []

    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error(f"[Evolver] JSON parse error: {e}")
        return []

    hypotheses = []
    valid_parent_ids = {parent.id for parent in parents}
    all_parent_ids = [parent.id for parent in parents]
    best_elo = max(parent.elo_rating for parent in parents) if parents else 1500
    parent_critiques = {
        critique
        for parent in parents
        for entry in parent.debate_history
        if (critique := entry.get("critique", ""))
    }

    for item in items:
        if not isinstance(item, dict):
            continue

        category_str = item.get("category", "integrative")
        try:
            category = HypothesisCategory(category_str)
        except ValueError:
            category = HypothesisCategory.INTEGRATIVE

        evolution_type = str(item.get("evolution_type", "strengthened")).lower()
        if evolution_type not in {"strengthened", "combined", "deepened", "divergent"}:
            evolution_type = "strengthened"

        requested_parent_ids = item.get("parent_ids", [])
        if not isinstance(requested_parent_ids, list):
            requested_parent_ids = []
        parent_ids = [parent_id for parent_id in requested_parent_ids if parent_id in valid_parent_ids]
        if not parent_ids and evolution_type != "divergent":
            parent_ids = all_parent_ids

        requested_critiques = item.get("addressed_critiques", [])
        if not isinstance(requested_critiques, list):
            requested_critiques = []
        addressed_critiques = [
            critique for critique in requested_critiques
            if isinstance(critique, str) and critique.strip()
        ]
        if not addressed_critiques and evolution_type != "divergent":
            addressed_critiques = list(parent_critiques)[:3]

        hypothesis = Hypothesis(
            condition=item.get("condition", ""),
            prediction=item.get("prediction", ""),
            mechanism=item.get("mechanism", ""),
            category=category,
            supporting_ptms=item.get("supporting_ptms", []),
            signaling_chain=item.get("signaling_chain", ""),
            testable_prediction=item.get("testable_prediction", ""),
            confidence=0.6,
            elo_rating=best_elo,
            status=HypothesisStatus.EVOLVED,
            parent_hypothesis_ids=parent_ids,
            evolution_type=evolution_type,
            addressed_critiques=addressed_critiques,
            generation_round=max(parent.generation_round for parent in parents) + 1,
        )

        if hypothesis.condition and hypothesis.prediction:
            hypotheses.append(hypothesis)

    return hypotheses

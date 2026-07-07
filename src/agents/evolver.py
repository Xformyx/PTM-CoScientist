"""
Evolver Agent — Refines and combines top-ranked hypotheses.

Inspired by Google Co-Scientist's Evolution agent.
Takes the highest-Elo hypotheses, addresses their weaknesses
(from debate critiques), and produces improved versions.
"""

import json
import logging
from typing import List, Dict, Any

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
- Address at least one critique from the debate
- Maintain testability — every hypothesis needs a clear experimental prediction

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
    "parent_ids": ["id1", "id2"]
  }
]
"""


def run_evolution(
    hypotheses: List[Hypothesis],
    llm: LLMClient,
    top_k: int = 3,
    context: Dict[str, Any] = None,
) -> List[Hypothesis]:
    """
    Evolve top-ranked hypotheses into improved versions.

    Args:
        hypotheses: Ranked hypotheses (from debater)
        llm: LLM client
        top_k: Number of top hypotheses to evolve from
        context: Additional context from PTM-platform

    Returns:
        New evolved hypotheses
    """
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


def _build_evolution_prompt(top_hypotheses: List[Hypothesis], context: Dict[str, Any] = None) -> str:
    """Build prompt for evolution."""
    parts = []

    parts.append("## Top-Ranked Hypotheses (from debate tournament)\n")
    for i, h in enumerate(top_hypotheses):
        parts.append(f"### Hypothesis {i+1} (Elo: {h.elo_rating}, ID: {h.id})")
        parts.append(f"- IF: {h.condition}")
        parts.append(f"- THEN: {h.prediction}")
        parts.append(f"- BECAUSE: {h.mechanism}")
        parts.append(f"- Signaling: {h.signaling_chain}")
        parts.append(f"- Supporting PTMs: {', '.join(h.supporting_ptms[:5])}")
        parts.append(f"- Confidence: {h.confidence}")
        parts.append(f"- Literature support: {len(h.evidence_for)} papers")

        # Include debate critiques
        critiques = [d.get("critique", "") for d in h.debate_history if d.get("critique")]
        if critiques:
            parts.append(f"- **Critiques to address:** {'; '.join(critiques[:3])}")
        parts.append("")

    # Additional context
    if context:
        kinase = context.get("kinase_modules", {})
        if kinase:
            parts.append(f"## Available Kinase Context\n{json.dumps(kinase, indent=2)[:1000]}")

    parts.append(f"\n## Task\nEvolve these {len(top_hypotheses)} hypotheses into 3-4 improved versions.")
    parts.append("At least one should COMBINE insights from multiple parent hypotheses.")
    parts.append("At least one should be a DIVERGENT angle not covered by any parent.")

    return "\n".join(parts)


def _parse_response(response: str, parents: List[Hypothesis]) -> List[Hypothesis]:
    """Parse evolved hypotheses from LLM response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
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
    parent_ids = [p.id for p in parents]
    # Inherit best parent's Elo as starting point
    best_elo = max(p.elo_rating for p in parents) if parents else 1500

    for item in items:
        if not isinstance(item, dict):
            continue

        category_str = item.get("category", "integrative")
        try:
            category = HypothesisCategory(category_str)
        except ValueError:
            category = HypothesisCategory.INTEGRATIVE

        h = Hypothesis(
            condition=item.get("condition", ""),
            prediction=item.get("prediction", ""),
            mechanism=item.get("mechanism", ""),
            category=category,
            supporting_ptms=item.get("supporting_ptms", []),
            signaling_chain=item.get("signaling_chain", ""),
            testable_prediction=item.get("testable_prediction", ""),
            confidence=0.6,  # Evolved hypotheses start with slightly higher confidence
            elo_rating=best_elo,  # Inherit best parent's Elo
            status=HypothesisStatus.EVOLVED,
            generation_round=max(p.generation_round for p in parents) + 1,
        )

        if h.condition and h.prediction:
            hypotheses.append(h)

    return hypotheses

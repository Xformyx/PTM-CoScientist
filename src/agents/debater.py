"""
Debater Agent — Critically evaluates and ranks hypotheses via tournament.

Inspired by Google Co-Scientist's Reflection + Ranking agents.
Performs pairwise comparisons using scientific debate, then updates Elo ratings.
Also validates hypotheses against ChromaDB literature evidence.
"""

import json
import logging
import itertools
from typing import List, Dict, Any, Optional, Tuple

from src.core.llm_client import LLMClient
from src.core.models import Hypothesis, HypothesisStatus
from src.core.elo import update_ratings, rank_hypotheses
from src.connectors.chromadb_connector import ChromaDBConnector

logger = logging.getLogger(__name__)

DEBATE_SYSTEM_PROMPT = """\
You are a critical scientific reviewer specializing in PTM biology and cell signaling.
You are evaluating two competing hypotheses in a scientific debate.

Your task: Compare the two hypotheses and determine which is STRONGER based on:
1. **Biological plausibility** — Is the mechanism well-grounded in known biology?
2. **Novelty** — Does it propose something genuinely new vs. restating known facts?
3. **Testability** — Can it be experimentally validated with current technology?
4. **Evidence grounding** — Is it supported by the PTM data provided?
5. **Specificity** — Is it precise enough to be falsifiable?

## OUTPUT FORMAT
Return ONLY a valid JSON object:
{
  "winner": "A" or "B" or "DRAW",
  "reasoning": "Brief explanation of why the winner is stronger",
  "critique_a": "Key weakness of Hypothesis A",
  "critique_b": "Key weakness of Hypothesis B"
}
"""

VALIDATION_SYSTEM_PROMPT = """\
You are a literature evidence classifier for PTM biology.
Given a hypothesis and a piece of literature evidence, classify the relationship.

Reply with ONLY one word: SUPPORTING, CONTRADICTING, or NEUTRAL
"""


def run_debate(
    hypotheses: List[Hypothesis],
    llm: LLMClient,
    chromadb: ChromaDBConnector,
    tournament_rounds: int = 3,
    k_factor: int = 32,
    rag_collections: Optional[List[str]] = None,
) -> List[Hypothesis]:
    """
    Run debate tournament on hypotheses.

    1. Validate each hypothesis against literature (ChromaDB)
    2. Run pairwise debate tournament
    3. Update Elo ratings

    Args:
        hypotheses: List of hypotheses to debate
        llm: LLM client
        chromadb: ChromaDB connector for evidence retrieval
        tournament_rounds: Number of tournament rounds
        k_factor: Elo K-factor
        rag_collections: ChromaDB collection names to restrict search (None = all)

    Returns:
        Hypotheses with updated Elo ratings and debate history
    """
    if len(hypotheses) < 2:
        return hypotheses

    # Step 1: Literature validation
    logger.info(f"[Debater] Validating {len(hypotheses)} hypotheses against literature")
    for h in hypotheses:
        _validate_against_literature(h, chromadb, llm, rag_collections=rag_collections)

    # Step 2: Pairwise tournament
    logger.info(f"[Debater] Running {tournament_rounds} tournament rounds")
    for round_num in range(tournament_rounds):
        pairs = _generate_matchups(hypotheses)
        for h_a, h_b in pairs:
            winner, critique = _debate_pair(h_a, h_b, llm)
            h_a.elo_rating, h_b.elo_rating = update_ratings(
                h_a.elo_rating, h_b.elo_rating, winner, k_factor
            )
            # Store debate history
            h_a.debate_history.append({
                "round": round_num,
                "opponent": h_b.id,
                "result": "win" if winner == "a" else ("loss" if winner == "b" else "draw"),
                "critique": critique.get("critique_a", ""),
            })
            h_b.debate_history.append({
                "round": round_num,
                "opponent": h_a.id,
                "result": "win" if winner == "b" else ("loss" if winner == "a" else "draw"),
                "critique": critique.get("critique_b", ""),
            })

    # Update status
    for h in hypotheses:
        h.status = HypothesisStatus.DEBATED

    ranked = rank_hypotheses(hypotheses)
    logger.info(f"[Debater] Tournament complete. Top Elo: {ranked[0].elo_rating}")
    return ranked


def _validate_against_literature(
    h: Hypothesis,
    chromadb: ChromaDBConnector,
    llm: LLMClient,
    rag_collections: Optional[List[str]] = None,
):
    """Validate a hypothesis against ChromaDB literature."""
    evidence = chromadb.search_for_hypothesis(h, collection_names=rag_collections)
    if not evidence:
        return

    for ev in evidence[:5]:
        doc = ev.get("document", "")
        if not doc:
            continue

        classification = _classify_evidence(doc, h, llm)
        entry = {
            "text": doc[:300],
            "source": ev.get("metadata", {}).get("title", "Unknown"),
            "collection": ev.get("collection", ""),
        }

        if classification == "supporting":
            h.evidence_for.append(entry)
        elif classification == "contradicting":
            h.evidence_against.append(entry)

    # Adjust confidence based on evidence
    support = len(h.evidence_for)
    contra = len(h.evidence_against)
    if support + contra > 0:
        h.confidence = round((support + 0.5) / (support + contra + 1), 2)


def _classify_evidence(evidence_text: str, hypothesis: Hypothesis, llm: LLMClient) -> str:
    """Classify evidence as supporting, contradicting, or neutral."""
    prompt = (
        f"Hypothesis: IF {hypothesis.condition} THEN {hypothesis.prediction}\n"
        f"Evidence: {evidence_text[:500]}\n\n"
        "Is this evidence SUPPORTING, CONTRADICTING, or NEUTRAL to the hypothesis?"
    )

    response = llm.generate(prompt, system_prompt=VALIDATION_SYSTEM_PROMPT, temperature=0.1, max_tokens=20)
    response = response.strip().upper()

    if "SUPPORT" in response:
        return "supporting"
    elif "CONTRADICT" in response:
        return "contradicting"
    return "neutral"


def _generate_matchups(hypotheses: List[Hypothesis]) -> List[Tuple[Hypothesis, Hypothesis]]:
    """Generate pairwise matchups for the tournament."""
    if len(hypotheses) <= 6:
        # Round-robin for small sets
        return list(itertools.combinations(hypotheses, 2))
    else:
        # Swiss-style: pair similar Elo ratings
        sorted_h = sorted(hypotheses, key=lambda h: h.elo_rating)
        pairs = []
        for i in range(0, len(sorted_h) - 1, 2):
            pairs.append((sorted_h[i], sorted_h[i + 1]))
        return pairs


def _debate_pair(h_a: Hypothesis, h_b: Hypothesis, llm: LLMClient) -> Tuple[str, Dict]:
    """Run a debate between two hypotheses."""
    prompt = f"""## Hypothesis A
IF: {h_a.condition}
THEN: {h_a.prediction}
BECAUSE: {h_a.mechanism}
Supporting PTMs: {', '.join(h_a.supporting_ptms[:5])}
Signaling chain: {h_a.signaling_chain}
Literature support: {len(h_a.evidence_for)} papers

## Hypothesis B
IF: {h_b.condition}
THEN: {h_b.prediction}
BECAUSE: {h_b.mechanism}
Supporting PTMs: {', '.join(h_b.supporting_ptms[:5])}
Signaling chain: {h_b.signaling_chain}
Literature support: {len(h_b.evidence_for)} papers

Which hypothesis is scientifically stronger?"""

    response = llm.generate(prompt, system_prompt=DEBATE_SYSTEM_PROMPT, temperature=0.3, max_tokens=512)

    # Parse response
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            result = json.loads(text[start:end + 1])
            winner_str = result.get("winner", "DRAW").upper()
            if winner_str == "A":
                return "a", result
            elif winner_str == "B":
                return "b", result
            else:
                return "draw", result
    except (json.JSONDecodeError, KeyError):
        pass

    return "draw", {"reasoning": "Parse error", "critique_a": "", "critique_b": ""}

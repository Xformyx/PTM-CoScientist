"""
Generator Agent — Proposes initial hypotheses from PTM-platform data.

Inspired by Google Co-Scientist's Generation + Proximity agents.
Uses enriched PTM data, kinase modules, and signal flow to generate
structured IF-THEN-BECAUSE hypotheses.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from src.core.llm_client import LLMClient
from src.core.models import Hypothesis, HypothesisCategory, HypothesisStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a PTM (Post-Translational Modification) research hypothesis generator.
You are part of an AI Co-Scientist system that helps researchers discover novel
biological mechanisms through PTM data analysis.

Your role: Generate novel, testable hypotheses based on enriched PTM data,
kinase-substrate relationships, temporal signaling patterns, and literature evidence.

## RULES
1. Each hypothesis MUST follow IF-THEN-BECAUSE format
2. Hypotheses must be grounded in the provided data (cite specific PTM sites)
3. Include a specific signaling chain (Receptor → Kinase → Substrate)
4. Prioritize novelty: propose connections not explicitly stated in the data
5. Each hypothesis must have a concrete testable prediction
6. Categorize as: mechanistic | temporal | predictive | integrative | therapeutic

## OUTPUT FORMAT
Return ONLY a valid JSON array (no markdown fences):
[
  {
    "condition": "IF ...",
    "prediction": "THEN ...",
    "mechanism": "BECAUSE ...",
    "category": "mechanistic",
    "supporting_ptms": ["GENE1-S123", "GENE2-T456"],
    "signaling_chain": "RECEPTOR → KINASE → SUBSTRATE",
    "testable_prediction": "Specific experiment that could confirm/refute this"
  }
]
"""


def run_generation(
    context: Dict[str, Any],
    llm: LLMClient,
    n_candidates: int = 5,
    research_goal: str = "",
    iteration: int = 0,
    chromadb=None,
    rag_collections: Optional[List[str]] = None,
) -> List[Hypothesis]:
    """
    Generate hypothesis candidates from PTM-platform context.

    Args:
        context: Assembled context from PTMPlatformConnector
        llm: LLM client instance
        n_candidates: Number of hypotheses to generate
        research_goal: User's research goal/question
        iteration: Current iteration (for diversity)
        chromadb: Optional ChromaDBConnector for live literature enrichment
        rag_collections: ChromaDB collection names to restrict search (None = all)

    Returns:
        List of Hypothesis objects
    """
    # Fetch relevant literature from ChromaDB to enrich the generation prompt
    lit_context: List[Dict[str, Any]] = []
    if chromadb is not None and chromadb.is_available():
        top_genes = [p.get("gene", "") for p in context.get("top_ptms", [])[:8] if p.get("gene")]
        ptm_type = context.get("ptm_type", "phosphorylation")
        lit_context = chromadb.search_for_context(
            genes=top_genes,
            ptm_type=ptm_type,
            collection_names=rag_collections,
            n_results=8,
        )
        logger.info(f"[Generator] Fetched {len(lit_context)} literature snippets from ChromaDB")

    user_prompt = _build_user_prompt(context, n_candidates, research_goal, iteration, lit_context)

    response = llm.generate(
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.6 + (iteration * 0.1),  # Increase creativity in later rounds
        max_tokens=4096,
    )

    hypotheses = _parse_response(response, iteration)
    logger.info(f"[Generator] Generated {len(hypotheses)} hypotheses (iteration {iteration})")
    return hypotheses


def _build_user_prompt(
    context: Dict[str, Any],
    n_candidates: int,
    research_goal: str,
    iteration: int,
    lit_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user prompt from context."""
    parts = []

    if research_goal:
        parts.append(f"## Research Goal\n{research_goal}")

    # Top PTMs
    top_ptms = context.get("top_ptms", [])
    if top_ptms:
        ptm_lines = []
        for p in top_ptms[:15]:
            fc = p.get("ptm_relative_log2fc", 0)
            direction = "↑" if fc > 0 else "↓"
            ptm_lines.append(
                f"- {p['gene']}-{p['position']} ({direction} Log2FC={fc:.2f}) "
                f"Pathways: {', '.join(str(pw) for pw in p.get('pathways', [])[:2])}"
            )
        parts.append("## Key PTM Sites\n" + "\n".join(ptm_lines))

    # Kinase modules
    kinase = context.get("kinase_modules", {})
    if kinase:
        modules = kinase.get("kinase_modules", [])
        if modules:
            km_lines = [f"- {m.get('kinase', 'Unknown')}: {len(m.get('substrates', []))} substrates" for m in modules[:5]]
            parts.append("## Active Kinase Modules\n" + "\n".join(km_lines))

    # Signal flow
    signal = context.get("signal_flow", {})
    if signal:
        parts.append(f"## Signal Flow Summary\n{json.dumps(signal, indent=2)[:1500]}")

    # Co-movement clusters
    clusters = context.get("comovement_clusters", {})
    if clusters:
        parts.append(f"## Temporal Co-movement\n{json.dumps(clusters, indent=2)[:1000]}")

    # Report excerpt
    report = context.get("comprehensive_report_excerpt", "")
    if report:
        parts.append(f"## Analysis Report Excerpt\n{report[:2000]}")

    # Live literature context from ChromaDB
    if lit_context:
        snippets = []
        seen_sources = set()
        for item in lit_context:
            source = item.get("metadata", {}).get("title", "Unknown")
            doc = item.get("document", "")
            if not doc or source in seen_sources:
                continue
            seen_sources.add(source)
            snippets.append(f"- [{source}] {doc[:250]}")
        if snippets:
            parts.append("## Relevant Literature (from ChromaDB)\n" + "\n".join(snippets[:6]))

    parts.append(f"\n## Task\nGenerate exactly {n_candidates} novel hypotheses.")
    if iteration > 0:
        parts.append(
            f"This is iteration {iteration}. Focus on NOVEL angles not covered in previous rounds. "
            "Explore cross-talk between pathways, temporal dependencies, or therapeutic implications."
        )

    return "\n\n".join(parts)


def _parse_response(response: str, iteration: int) -> List[Hypothesis]:
    """Parse LLM response into Hypothesis objects."""
    # Try to extract JSON from response
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[Generator] Could not find JSON array in response")
        return []

    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error(f"[Generator] JSON parse error: {e}")
        return []

    hypotheses = []
    for item in items:
        if not isinstance(item, dict):
            continue

        category_str = item.get("category", "mechanistic")
        try:
            category = HypothesisCategory(category_str)
        except ValueError:
            category = HypothesisCategory.MECHANISTIC

        h = Hypothesis(
            condition=item.get("condition", ""),
            prediction=item.get("prediction", ""),
            mechanism=item.get("mechanism", ""),
            category=category,
            supporting_ptms=item.get("supporting_ptms", []),
            signaling_chain=item.get("signaling_chain", ""),
            testable_prediction=item.get("testable_prediction", ""),
            confidence=0.5,
            status=HypothesisStatus.GENERATED,
            generation_round=iteration,
        )

        if h.condition and h.prediction:
            hypotheses.append(h)

    return hypotheses

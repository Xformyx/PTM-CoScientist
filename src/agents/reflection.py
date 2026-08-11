"""Reflection Agent — self-critique before peer debate.

The agent turns a hypothesis into an auditable set of atomic claims and checks
its consistency with measured PTM observations, retrieved literature, graph
neighbours, possible confounders, and falsification conditions. It does not
claim to validate a hypothesis; it records what must be scrutinised next.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.evidence_graph import graph_neighborhood
from src.core.llm_client import LLMClient
from src.core.models import Hypothesis, HypothesisStatus, LabResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a rigorous PTM biology reviewer performing self-critique of a proposed
hypothesis before it enters peer debate. You must distinguish measured facts,
literature-supported claims, and untested interpretation. Do not invent data or
citations beyond the supplied context.

Review for: atomic claim structure, consistency with observed PTM sites and
temporal context, support versus contradiction in literature, novelty relative
to supplied literature, plausible confounders, missing evidence, and concrete
falsification conditions.

Return ONLY a JSON object:
{
  "atomic_claims": [{"claim": "...", "evidence_status": "observed|literature_supported|untested"}],
  "data_consistency": "consistent|partial|inconsistent|insufficient",
  "literature_consistency": "supported|mixed|contradicted|insufficient",
  "novelty_assessment": "novel_connection|incremental|already_established|uncertain",
  "confounders": ["..."],
  "missing_evidence": ["..."],
  "falsification_conditions": ["..."],
  "recommended_action": "advance|revise|deprioritize",
  "summary": "..."
}
"""


def run_reflection(
    hypotheses: list[Hypothesis],
    *,
    context: dict[str, Any],
    evidence_graph: dict[str, Any],
    lab_results: list[LabResult],
    llm: LLMClient,
) -> list[Hypothesis]:
    """Attach a structured self-critique to every candidate hypothesis."""
    for hypothesis in hypotheses:
        prompt = _build_prompt(hypothesis, context, evidence_graph, lab_results)
        response = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.15,
            max_tokens=1800,
        )
        reflection = _parse_reflection(response)
        hypothesis.reflection = reflection
        if hypothesis.status == HypothesisStatus.GENERATED:
            hypothesis.status = HypothesisStatus.REFLECTED
    logger.info("[Reflection] Reviewed %d hypotheses", len(hypotheses))
    return hypotheses


def _build_prompt(
    hypothesis: Hypothesis,
    context: dict[str, Any],
    evidence_graph: dict[str, Any],
    lab_results: list[LabResult],
) -> str:
    measured = _matching_measurements(hypothesis.supporting_ptms, context.get("top_ptms", []))
    graph_edges = graph_neighborhood(evidence_graph, hypothesis.supporting_ptms, max_edges=12)
    associated_results = [result.to_dict() for result in lab_results if result.hypothesis_id == hypothesis.id]

    parts = [
        "## Research Goal",
        context.get("research_goal", "No explicit research goal supplied."),
        "\n## Candidate Hypothesis",
        f"IF: {hypothesis.condition}",
        f"THEN: {hypothesis.prediction}",
        f"BECAUSE: {hypothesis.mechanism}",
        f"Signaling chain: {hypothesis.signaling_chain}",
        f"Supporting PTM sites: {', '.join(hypothesis.supporting_ptms)}",
        f"Testable prediction: {hypothesis.testable_prediction}",
        "\n## Matched Measured PTM Observations",
        json.dumps(measured, ensure_ascii=False)[:3000],
        "\n## Evidence Graph Neighbourhood",
        json.dumps(graph_edges, ensure_ascii=False)[:3500],
        "\n## Retrieved Supporting Literature",
        json.dumps(_compact_evidence(hypothesis.evidence_for), ensure_ascii=False)[:2500],
        "\n## Retrieved Contradicting Literature",
        json.dumps(_compact_evidence(hypothesis.evidence_against), ensure_ascii=False)[:2500],
        "\n## Prior Laboratory Results",
        json.dumps(associated_results, ensure_ascii=False)[:2000],
    ]
    return "\n".join(parts)


def _matching_measurements(sites: list[str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    wanted = {str(site).upper() for site in sites}
    for record in records or []:
        gene = str(record.get("gene") or record.get("gene_name") or record.get("Gene_Name") or "")
        position = str(record.get("position") or record.get("Position") or "")
        site = f"{gene}-{position}".upper() if position else gene.upper()
        if site in wanted or gene.upper() in wanted:
            result.append({
                "site": site,
                "condition": record.get("condition") or record.get("timepoint") or "",
                "ptm_relative_log2fc": record.get("ptm_relative_log2fc", record.get("fold_change")),
                "protein_log2fc": record.get("protein_log2fc"),
                "pathways": record.get("pathways", []),
            })
    return result


def _compact_evidence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": entry.get("title") or entry.get("source") or "",
            "pmid": entry.get("pmid") or "",
            "doi": entry.get("doi") or "",
            "excerpt": str(entry.get("excerpt") or entry.get("text") or "")[:400],
        }
        for entry in entries[:5]
    ]


def _parse_reflection(response: str) -> dict[str, Any]:
    """Parse model output and return a conservative structured fallback on failure."""
    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            payload = json.loads(text[start:end + 1])
            if isinstance(payload, dict):
                return _normalise_reflection(payload)
        except json.JSONDecodeError:
            pass
    logger.warning("[Reflection] Could not parse reflection JSON")
    return {
        "atomic_claims": [],
        "data_consistency": "insufficient",
        "literature_consistency": "insufficient",
        "novelty_assessment": "uncertain",
        "confounders": ["Reflection parsing failed; manual review required."],
        "missing_evidence": ["Structured reflection unavailable."],
        "falsification_conditions": [],
        "recommended_action": "revise",
        "summary": "Reflection unavailable due to malformed model response.",
    }


def _normalise_reflection(payload: dict[str, Any]) -> dict[str, Any]:
    def safe_list(key: str) -> list[Any]:
        value = payload.get(key, [])
        return value if isinstance(value, list) else []

    choices = {
        "data_consistency": {"consistent", "partial", "inconsistent", "insufficient"},
        "literature_consistency": {"supported", "mixed", "contradicted", "insufficient"},
        "novelty_assessment": {"novel_connection", "incremental", "already_established", "uncertain"},
        "recommended_action": {"advance", "revise", "deprioritize"},
    }
    normalized = {
        "atomic_claims": [item for item in safe_list("atomic_claims") if isinstance(item, dict)],
        "confounders": [str(item) for item in safe_list("confounders") if str(item).strip()],
        "missing_evidence": [str(item) for item in safe_list("missing_evidence") if str(item).strip()],
        "falsification_conditions": [str(item) for item in safe_list("falsification_conditions") if str(item).strip()],
        "summary": str(payload.get("summary") or ""),
    }
    for key, allowed in choices.items():
        value = str(payload.get(key) or "").lower()
        normalized[key] = value if value in allowed else "insufficient" if key != "recommended_action" else "revise"
    return normalized

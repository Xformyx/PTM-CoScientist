"""Build a report-safe Discussion Evidence Packet from Co-Scientist state.

The packet deliberately separates observed data support from AI-generated
interpretations. It is designed for a downstream PTM-platform writer to use as
*interpretive context*, never as a replacement for measured results.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from src.core.models import CoScientistState, Hypothesis

SCHEMA_VERSION = "1.0"
MAX_PACKET_HYPOTHESES = 5


def build_discussion_evidence_packet(
    state: CoScientistState,
    *,
    session_id: str,
    source_orders: list[str] | None = None,
    created_at: str | None = None,
    max_hypotheses: int = 3,
) -> dict[str, Any]:
    """Create a bounded, evidence-gated packet for a report Discussion.

    Only candidates meeting all gates are returned in ``selected_hypotheses``:
    a structured claim, at least one measured PTM data observation, at least one
    citable supporting literature item, and either counter-evidence or a debate
    limitation. Elo is intentionally converted into an internal priority tier;
    its numeric value is never exported as a scientific confidence metric.
    """
    max_hypotheses = max(1, min(int(max_hypotheses), MAX_PACKET_HYPOTHESES))
    generated_at = created_at or datetime.now(UTC).isoformat()
    source_orders = source_orders or []

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    ranked = sorted(state.hypotheses, key=lambda hypothesis: hypothesis.elo_rating, reverse=True)

    for hypothesis in ranked:
        candidate, gate = _build_candidate(hypothesis, state)
        if gate["passed"] and len(selected) < max_hypotheses:
            candidate["priority_tier"] = _priority_tier(len(selected))
            selected.append(candidate)
        else:
            excluded.append({
                "hypothesis_id": hypothesis.id,
                "reason": gate["reason"],
            })

    selected_ids = {candidate["id"] for candidate in selected}
    experiment_priorities = [
        _experiment_to_packet(design.to_dict())
        for design in state.experiment_designs
        if design.hypothesis_id in selected_ids
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "packet_type": "discussion_evidence_packet",
        "session_id": session_id,
        "generated_at": generated_at,
        "source_orders": source_orders,
        "research_goal": state.research_goal,
        "ptm_type": state.experimental_context.get("ptm_type", ""),
        "rag_collections": state.rag_collections,
        "status": "ready" if selected else "no_eligible_hypotheses",
        "usage_notice": (
            "Use selected hypotheses only as interpretive, falsifiable candidates in "
            "a Discussion. Do not present them as measured findings, causal proof, "
            "or statistically validated conclusions. Verify bibliography and cite "
            "primary sources through PTM-platform before publication."
        ),
        "selected_hypotheses": selected,
        "experiment_priorities": experiment_priorities,
        "quality_summary": {
            "evaluated_hypotheses": len(ranked),
            "eligible_hypotheses": len(selected),
            "excluded_candidates": excluded,
            "max_hypotheses": max_hypotheses,
        },
    }


def _build_candidate(hypothesis: Hypothesis, state: CoScientistState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one candidate and evaluate the report-safety quality gates."""
    data_support = _data_support(hypothesis.supporting_ptms, state.enriched_ptm_data)
    literature_evidence = _normalize_evidence(hypothesis.evidence_for, "supporting")
    counter_evidence = _normalize_evidence(hypothesis.evidence_against, "contradicting")
    limitations = _collect_limitations(hypothesis, counter_evidence)

    claim = _hypothesis_claim(hypothesis)
    has_structured_claim = bool(hypothesis.condition and hypothesis.prediction and hypothesis.mechanism)
    has_data_support = bool(data_support)
    has_citable_support = any(_is_citable(evidence) for evidence in literature_evidence)
    has_uncertainty = bool(counter_evidence or limitations)

    missing = []
    if not has_structured_claim:
        missing.append("missing_if_then_because_claim")
    if not has_data_support:
        missing.append("missing_measured_ptm_data_support")
    if not has_citable_support:
        missing.append("missing_citable_supporting_literature")
    if not has_uncertainty:
        missing.append("missing_counter_evidence_or_debate_limitation")

    candidate = {
        "id": hypothesis.id,
        "priority_tier": "",  # assigned after eligible ranking
        "claim": claim,
        "category": hypothesis.category.value,
        "supporting_ptm_sites": hypothesis.supporting_ptms,
        "signaling_chain": hypothesis.signaling_chain,
        "data_support": data_support,
        "literature_evidence": literature_evidence,
        "counter_evidence": counter_evidence,
        "limitations": limitations,
        "testable_prediction": hypothesis.testable_prediction,
        "lineage": {
            "parent_hypothesis_ids": hypothesis.parent_hypothesis_ids,
            "evolution_type": hypothesis.evolution_type,
            "addressed_critiques": hypothesis.addressed_critiques,
        },
        "quality_gate": {
            "passed": not missing,
            "reason": "eligible" if not missing else ";".join(missing),
            "requirements": {
                "structured_claim": has_structured_claim,
                "measured_ptm_data_support": has_data_support,
                "citable_supporting_literature": has_citable_support,
                "counter_evidence_or_limitation": has_uncertainty,
            },
        },
    }
    return candidate, candidate["quality_gate"]


def _hypothesis_claim(hypothesis: Hypothesis) -> str:
    """Render a clearly labelled hypothesis rather than a factual claim."""
    return " ".join(
        part.strip()
        for part in (hypothesis.condition, hypothesis.prediction, hypothesis.mechanism)
        if part and part.strip()
    )


def _data_support(sites: Iterable[str], enriched_ptm_data: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map stated supporting sites to the measured PTM records available in state."""
    records = list(enriched_ptm_data or [])
    output: list[dict[str, Any]] = []
    seen = set()

    for site in sites or []:
        gene, position = _split_site(site)
        if not gene:
            continue
        for record in records:
            record_gene = str(record.get("gene") or record.get("gene_name") or record.get("Gene_Name") or "")
            record_position = str(record.get("position") or record.get("Position") or "")
            if record_gene != gene:
                continue
            if position and record_position and position != record_position:
                continue
            key = (record_gene, record_position, record.get("condition", ""))
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "source": "PTM-platform measured data",
                "gene": record_gene,
                "position": record_position,
                "condition": record.get("condition") or record.get("timepoint") or "",
                "ptm_relative_log2fc": record.get("ptm_relative_log2fc", record.get("fold_change")),
                "protein_log2fc": record.get("protein_log2fc", ""),
                "pathways": record.get("pathways", []),
            })
    return output


def _split_site(site: str) -> tuple[str, str]:
    """Split conventional ``GENE-S123`` notation without imposing hard-coded genes."""
    if not site or "-" not in site:
        return str(site or ""), ""
    gene, position = str(site).rsplit("-", 1)
    return gene.strip(), position.strip()


def _normalize_evidence(entries: Iterable[dict[str, Any]], stance: str) -> list[dict[str, Any]]:
    """Convert legacy and current evidence records into a citation-oriented schema."""
    output = []
    seen = set()
    for entry in entries or []:
        title = str(entry.get("title") or entry.get("source") or "")
        evidence_id = str(entry.get("evidence_id") or entry.get("id") or "")
        pmid = str(entry.get("pmid") or entry.get("pubmed_id") or "")
        doi = str(entry.get("doi") or "")
        key = (evidence_id, pmid, doi, title)
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "evidence_id": evidence_id,
            "stance": stance,
            "title": title,
            "pmid": pmid,
            "doi": doi,
            "authors": str(entry.get("authors") or ""),
            "year": str(entry.get("year") or ""),
            "journal": str(entry.get("journal") or ""),
            "collection": str(entry.get("collection") or ""),
            "retrieval_score": entry.get("retrieval_score"),
            "excerpt": str(entry.get("excerpt") or entry.get("text") or ""),
        })
    return output


def _collect_limitations(hypothesis: Hypothesis, counter_evidence: list[dict[str, Any]]) -> list[str]:
    """Collect explicit uncertainty from debate criticism and counter-evidence."""
    limitations = []
    for item in hypothesis.debate_history:
        critique = str(item.get("critique") or "").strip()
        if critique and critique not in limitations:
            limitations.append(critique)
    for item in counter_evidence:
        title = item.get("title") or "retrieved literature"
        text = item.get("excerpt") or ""
        limitation = f"Contradicting evidence from {title}: {text}".strip()
        if limitation not in limitations:
            limitations.append(limitation)
    return limitations[:5]


def _is_citable(evidence: dict[str, Any]) -> bool:
    """Require a title and a durable identifier before a claim enters a report packet."""
    return bool(evidence.get("title") and (evidence.get("pmid") or evidence.get("doi") or evidence.get("evidence_id")))


def _priority_tier(index: int) -> str:
    return ("high", "medium", "exploratory")[min(index, 2)]


def _experiment_to_packet(design: dict[str, Any]) -> dict[str, Any]:
    """Expose only the fields appropriate for a future-validation section."""
    return {
        "hypothesis_id": design.get("hypothesis_id", ""),
        "title": design.get("title", ""),
        "objective": design.get("objective", ""),
        "approach": design.get("approach", ""),
        "key_reagents": design.get("key_reagents", []),
        "controls": design.get("controls", []),
        "expected_outcome": design.get("expected_outcome", ""),
        "alternative_outcome": design.get("alternative_outcome", ""),
        "estimated_timeline": design.get("estimated_timeline", ""),
        "priority": design.get("priority", "medium"),
        "rationale": design.get("rationale", ""),
    }

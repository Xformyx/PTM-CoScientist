"""Minimal, portable PTM Evidence Graph for scientific reasoning.

The graph is intentionally a versioned JSON structure rather than a new graph
server. It remains bounded to PTM-platform artifacts, existing ChromaDB
literature metadata, Co-Scientist hypotheses, and researcher-entered lab
results. This preserves read-only integration and makes provenance auditable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.core.models import Hypothesis, LabResult

SCHEMA_VERSION = "1.0"


def build_evidence_graph(
    context: dict[str, Any],
    hypotheses: Iterable[Hypothesis] | None = None,
    lab_results: Iterable[LabResult] | None = None,
) -> dict[str, Any]:
    """Build a JSON evidence graph from available PTM and RAG-derived context."""
    graph: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_type": "ptm_evidence_graph",
        "nodes": [],
        "edges": [],
        "summary": {},
    }
    node_ids: set[str] = set()
    edge_ids: set[str] = set()

    def add_node(node_id: str, node_type: str, label: str, **attributes: Any) -> str:
        if not node_id or node_id in node_ids:
            return node_id
        node_ids.add(node_id)
        graph["nodes"].append({
            "id": node_id,
            "type": node_type,
            "label": label,
            "attributes": _clean(attributes),
        })
        return node_id

    def add_edge(
        source: str,
        target: str,
        relation: str,
        *,
        provenance: dict[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        if not source or not target or not relation:
            return
        edge_id = f"{source}|{relation}|{target}"
        if edge_id in edge_ids:
            return
        edge_ids.add(edge_id)
        graph["edges"].append({
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "provenance": _clean(provenance or {}),
            "attributes": _clean(attributes),
        })

    ptm_type = str(context.get("ptm_type") or "")
    order_codes = context.get("order_codes") or ([context["order_code"]] if context.get("order_code") else [])
    for order_code in order_codes:
        add_node(f"order:{order_code}", "Order", str(order_code), source="PTM-platform")

    treatment = _first_nonempty(context, "treatment", "stimulus", "condition")
    if treatment:
        add_node(f"treatment:{treatment}", "Treatment", treatment, source="PTM-platform")

    # Observed PTM sites and their pathway membership.
    for record in context.get("top_ptms", []) or []:
        gene = _first_nonempty(record, "gene", "gene_name", "Gene_Name", "Gene.Name")
        position = _first_nonempty(record, "position", "Position")
        if not gene:
            continue
        protein_id = f"protein:{gene}"
        site_label = f"{gene}-{position}" if position else gene
        site_id = f"ptm_site:{site_label}"
        add_node(protein_id, "Protein", gene, source="PTM-platform")
        add_node(
            site_id,
            "PTMSite",
            site_label,
            ptm_type=record.get("ptm_type", ptm_type),
            ptm_relative_log2fc=record.get("ptm_relative_log2fc", record.get("fold_change")),
            protein_log2fc=record.get("protein_log2fc"),
            condition=record.get("condition") or record.get("timepoint") or "",
            order_codes=record.get("order_codes", order_codes),
            source="PTM-platform measured data",
        )
        add_edge(
            protein_id,
            site_id,
            "HAS_PTM_SITE",
            provenance={"type": "measured_ptm", "orders": record.get("order_codes", order_codes)},
        )
        for order_code in record.get("order_codes", order_codes) or []:
            add_edge(
                site_id,
                f"order:{order_code}",
                "OBSERVED_IN",
                provenance={"type": "measured_ptm"},
            )
        for pathway in _as_list(record.get("pathways")):
            pathway_id = f"pathway:{pathway}"
            add_node(pathway_id, "Pathway", str(pathway), source="PTM-platform RAG enrichment")
            add_edge(site_id, pathway_id, "PART_OF_PATHWAY", provenance={"type": "rag_enrichment"})

    # Kinase/E3 module relationships. The normalizer accepts dictionaries,
    # strings, and legacy list shapes without hard-coding a specific PTM type.
    for module in _as_list((context.get("kinase_modules") or {}).get("kinase_modules")):
        if not isinstance(module, dict):
            continue
        regulator = _first_nonempty(module, "kinase", "e3_ligase", "regulator", "gene")
        if not regulator:
            continue
        regulator_id = f"protein:{regulator}"
        add_node(regulator_id, "Regulator", regulator, source="PTM-platform kinase module")
        for substrate in _as_list(module.get("substrates")):
            substrate_gene, substrate_site = _normalise_substrate(substrate)
            if not substrate_gene:
                continue
            target_id = f"ptm_site:{substrate_gene}-{substrate_site}" if substrate_site else f"protein:{substrate_gene}"
            add_node(
                target_id,
                "PTMSite" if substrate_site else "Protein",
                f"{substrate_gene}-{substrate_site}" if substrate_site else substrate_gene,
                source="PTM-platform kinase module",
            )
            add_edge(
                regulator_id,
                target_id,
                "MODIFIES",
                provenance={"type": "kinase_module", "order_code": module.get("order_code", "")},
            )

    # Context-provided signal flow is preserved as edges when it already uses a
    # conventional edge shape. Unknown nested data is not hallucinated into graph relations.
    for edge in _extract_declared_edges(context.get("signal_flow", {})):
        source_label = edge["source"]
        target_label = edge["target"]
        source_id = f"protein:{source_label}"
        target_id = f"protein:{target_label}"
        add_node(source_id, "Protein", source_label, source="PTM-platform signal flow")
        add_node(target_id, "Protein", target_label, source="PTM-platform signal flow")
        add_edge(
            source_id,
            target_id,
            edge["relation"],
            provenance={"type": "signal_flow"},
            timepoint=edge.get("timepoint", ""),
        )

    # Temporal co-movement becomes a cluster-membership relation without
    # incorrectly asserting causal directionality.
    for cluster_name, members in _iter_clusters(context.get("comovement_clusters", {})):
        cluster_id = f"cluster:{cluster_name}"
        add_node(cluster_id, "TemporalCluster", cluster_name, source="PTM-platform co-movement")
        for member in members:
            gene, position = _normalise_substrate(member)
            if not gene:
                continue
            member_id = f"ptm_site:{gene}-{position}" if position else f"protein:{gene}"
            add_node(member_id, "PTMSite" if position else "Protein", f"{gene}-{position}" if position else gene)
            add_edge(member_id, cluster_id, "CO_MOVES_WITH", provenance={"type": "temporal_co_movement"})

    # Hypothesis, literature, and lab-result provenance are added after the
    # measured graph so every generated claim has inspectable neighbours.
    for hypothesis in hypotheses or []:
        hypothesis_id = f"hypothesis:{hypothesis.id}"
        add_node(
            hypothesis_id,
            "Hypothesis",
            hypothesis.prediction or hypothesis.id,
            category=hypothesis.category.value,
            status=hypothesis.status.value,
            generation_round=hypothesis.generation_round,
        )
        for site in hypothesis.supporting_ptms:
            gene, position = _normalise_substrate(site)
            if gene:
                site_id = f"ptm_site:{gene}-{position}" if position else f"protein:{gene}"
                add_edge(site_id, hypothesis_id, "SUPPORTS", provenance={"type": "hypothesis_data_link"})
        _add_literature_edges(add_node, add_edge, hypothesis_id, hypothesis.evidence_for, "SUPPORTS")
        _add_literature_edges(add_node, add_edge, hypothesis_id, hypothesis.evidence_against, "CONTRADICTS")

    for result in lab_results or []:
        result_id = f"lab_result:{result.id}"
        hypothesis_id = f"hypothesis:{result.hypothesis_id}"
        add_node(
            result_id,
            "LabResult",
            result.assay_type or result.id,
            outcome=result.outcome,
            result_summary=result.result_summary,
            observed_effect=result.observed_effect,
            controls=result.controls,
            source_reference=result.source_reference,
        )
        relation = {"supports": "SUPPORTS", "contradicts": "CONTRADICTS"}.get(result.outcome, "TESTS")
        add_edge(result_id, hypothesis_id, relation, provenance={"type": "researcher_entered_lab_result"})

    graph["summary"] = graph_summary(graph)
    return graph


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """Return compact counts for prompts, UI, and audit logs."""
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for node in graph.get("nodes", []):
        node_type = str(node.get("type", "Unknown"))
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    for edge in graph.get("edges", []):
        relation = str(edge.get("relation", "UNKNOWN"))
        edge_counts[relation] = edge_counts.get(relation, 0) + 1
    return {
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "node_types": node_counts,
        "relations": edge_counts,
    }


def graph_neighborhood(graph: dict[str, Any], seed_labels: Iterable[str], max_edges: int = 12) -> list[dict[str, Any]]:
    """Return provenance-preserving edges touching supplied genes or PTM labels."""
    labels = {str(label).lower() for label in seed_labels if label}
    if not labels:
        return []
    matched_nodes = {
        node["id"]
        for node in graph.get("nodes", [])
        if str(node.get("label", "")).lower() in labels
        or any(label in str(node.get("label", "")).lower() for label in labels)
    }
    return [
        edge for edge in graph.get("edges", [])
        if edge.get("source") in matched_nodes or edge.get("target") in matched_nodes
    ][:max_edges]


def _add_literature_edges(add_node, add_edge, hypothesis_id: str, entries: Iterable[dict[str, Any]], relation: str) -> None:
    for entry in entries or []:
        evidence_id = str(entry.get("evidence_id") or entry.get("id") or "")
        pmid = str(entry.get("pmid") or "")
        doi = str(entry.get("doi") or "")
        title = str(entry.get("title") or entry.get("source") or "")
        paper_key = pmid or doi or evidence_id or title
        if not paper_key:
            continue
        paper_id = f"paper:{paper_key}"
        add_node(
            paper_id,
            "Paper",
            title or paper_key,
            pmid=pmid,
            doi=doi,
            journal=entry.get("journal", ""),
            year=entry.get("year", ""),
            collection=entry.get("collection", ""),
        )
        add_edge(
            paper_id,
            hypothesis_id,
            relation,
            provenance={
                "type": "chromadb_literature",
                "evidence_id": evidence_id,
                "excerpt": str(entry.get("excerpt") or entry.get("text") or "")[:400],
            },
        )


def _extract_declared_edges(payload: Any) -> list[dict[str, str]]:
    """Extract only explicitly declared edges from heterogeneous signal-flow JSON."""
    edges: list[dict[str, str]] = []
    if isinstance(payload, dict):
        raw_edges = payload.get("edges") or payload.get("relationships") or payload.get("links") or []
        if isinstance(raw_edges, list):
            for item in raw_edges:
                if not isinstance(item, dict):
                    continue
                source = _first_nonempty(item, "source", "from", "upstream", "regulator")
                target = _first_nonempty(item, "target", "to", "downstream", "substrate")
                if source and target:
                    edges.append({
                        "source": source,
                        "target": target,
                        "relation": _first_nonempty(item, "relation", "type", "interaction") or "REGULATES",
                        "timepoint": _first_nonempty(item, "timepoint", "time"),
                    })
    return edges


def _iter_clusters(payload: Any) -> list[tuple[str, list[Any]]]:
    """Normalise common co-movement cluster shapes without creating unknown edges."""
    if not isinstance(payload, dict):
        return []
    raw_clusters = payload.get("clusters") or payload.get("groups") or payload
    output: list[tuple[str, list[Any]]] = []
    if isinstance(raw_clusters, dict):
        for name, value in raw_clusters.items():
            if isinstance(value, dict):
                members = value.get("members") or value.get("sites") or value.get("proteins") or []
            else:
                members = value
            if isinstance(members, list):
                output.append((str(name), members))
    elif isinstance(raw_clusters, list):
        for index, value in enumerate(raw_clusters):
            if not isinstance(value, dict):
                continue
            name = str(value.get("name") or value.get("id") or f"cluster_{index + 1}")
            members = value.get("members") or value.get("sites") or value.get("proteins") or []
            if isinstance(members, list):
                output.append((name, members))
    return output


def _normalise_substrate(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        return (
            _first_nonempty(value, "gene", "protein", "substrate", "name"),
            _first_nonempty(value, "position", "site", "residue"),
        )
    text = str(value or "").strip()
    if not text:
        return "", ""
    if "-" in text:
        gene, position = text.rsplit("-", 1)
        return gene.strip(), position.strip()
    return text, ""


def _first_nonempty(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _clean(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

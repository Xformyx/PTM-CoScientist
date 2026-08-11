"""Proximity Agent — deterministic diversity-aware hypothesis selection.

Google's Proximity role maps and clusters related ideas before final selection.
This module uses transparent feature overlap rather than an opaque secondary LLM:
PTM sites, signalling-chain tokens, categories, and meaningful claim tokens.
It assigns clusters but never discards candidates; consumers can inspect all
members and choose one representative per cluster for a diverse shortlist.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.models import Hypothesis

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_STOPWORDS = {
    "then", "because", "with", "from", "that", "this", "will", "when",
    "where", "which", "into", "than", "have", "has", "are", "for", "and",
    "the", "its", "via", "may", "could", "should", "under", "after", "before",
}


def cluster_and_select_diverse_hypotheses(
    hypotheses: list[Hypothesis],
    *,
    max_hypotheses: int = 5,
    similarity_threshold: float = 0.58,
) -> tuple[list[Hypothesis], dict[str, Any]]:
    """Cluster candidates by transparent overlap and choose diverse representatives."""
    ranked = sorted(hypotheses, key=lambda hypothesis: hypothesis.elo_rating, reverse=True)
    if not ranked:
        return [], {
            "method": "feature_jaccard_v1",
            "cluster_count": 0,
            "recommended_hypothesis_ids": [],
            "clusters": [],
        }

    clusters: list[list[Hypothesis]] = []
    feature_sets: dict[str, set[str]] = {hypothesis.id: _features(hypothesis) for hypothesis in ranked}
    for hypothesis in ranked:
        assigned = False
        for members in clusters:
            representative = members[0]
            similarity = _similarity(feature_sets[hypothesis.id], feature_sets[representative.id])
            if similarity >= similarity_threshold:
                members.append(hypothesis)
                assigned = True
                break
        if not assigned:
            clusters.append([hypothesis])

    for index, members in enumerate(clusters, start=1):
        cluster_id = f"proximity_{index:02d}"
        for member in members:
            member.proximity_cluster = cluster_id

    # Select one ranked representative from each cluster, then use additional
    # ranked cluster members only if space remains.
    representatives = [members[0] for members in clusters]
    selected = representatives[:max_hypotheses]
    if len(selected) < max_hypotheses:
        selected_ids = {hypothesis.id for hypothesis in selected}
        for hypothesis in ranked:
            if hypothesis.id not in selected_ids:
                selected.append(hypothesis)
                selected_ids.add(hypothesis.id)
            if len(selected) >= max_hypotheses:
                break

    summary = {
        "method": "feature_jaccard_v1",
        "similarity_threshold": similarity_threshold,
        "cluster_count": len(clusters),
        "recommended_hypothesis_ids": [hypothesis.id for hypothesis in selected],
        "clusters": [
            {
                "id": f"proximity_{index:02d}",
                "representative_hypothesis_id": members[0].id,
                "member_hypothesis_ids": [member.id for member in members],
            }
            for index, members in enumerate(clusters, start=1)
        ],
    }
    return selected, summary


def _features(hypothesis: Hypothesis) -> set[str]:
    tokens = {
        f"site:{site.lower()}"
        for site in hypothesis.supporting_ptms
        if str(site).strip()
    }
    if hypothesis.category:
        tokens.add(f"category:{hypothesis.category.value}")
    claim_text = (
        f"{hypothesis.condition} {hypothesis.prediction} "
        f"{hypothesis.mechanism} {hypothesis.signaling_chain}"
    )
    for token in _TOKEN_RE.findall(claim_text):
        normalized = token.lower()
        if normalized not in _STOPWORDS and len(normalized) > 2:
            tokens.add(f"token:{normalized}")
    return tokens


def _similarity(left: set[str], right: set[str]) -> float:
    """Compute transparent similarity while prioritising PTM-site overlap.

    Mechanistic hypotheses often share generic signalling terms such as receptor
    or kinase names. If both candidates name concrete but different PTM sites,
    lexical overlap alone must not collapse them into one proximity cluster.
    """
    left_sites = {feature for feature in left if feature.startswith("site:")}
    right_sites = {feature for feature in right if feature.startswith("site:")}
    text_left = left - left_sites
    text_right = right - right_sites
    text_score = _jaccard(text_left, text_right)
    if left_sites and right_sites:
        site_score = _jaccard(left_sites, right_sites)
        return (0.8 * site_score) + (0.2 * text_score)
    return text_score


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0

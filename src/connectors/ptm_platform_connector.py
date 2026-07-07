"""
PTM-Platform Connector — Read-only access to PTM-platform's artifacts.

Reads enriched PTM data (JSON files), order metadata (MySQL),
and cached analysis results (kinase modules, signal flow, etc.)
without modifying any PTM-platform data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PTMPlatformConnector:
    """
    Read-only connector to PTM-platform's output artifacts.

    Supports two modes:
    1. File-based: reads from mounted volume (enriched JSON, MD reports)
    2. DB-based: reads from MySQL (order metadata, cached analysis)
    """

    def __init__(self, artifacts_dir: str = "/data/ptm-platform/outputs", database_url: str = ""):
        self.artifacts_dir = Path(artifacts_dir)
        self.database_url = database_url
        self._db_engine = None

    # ─── File-based artifact reading ─────────────────────────────────────

    def load_enriched_ptm_data(self, order_code: str, ptm_type: str = "phosphorylation") -> List[Dict[str, Any]]:
        """Load enriched PTM data JSON for a given order."""
        suffix = "_phospho" if ptm_type == "phosphorylation" else "_ubi"
        json_path = self.artifacts_dir / order_code / f"enriched_ptm_data{suffix}.json"

        if not json_path.exists():
            # Try alternative naming
            alt_path = self.artifacts_dir / order_code / "enriched_ptm_data.json"
            if alt_path.exists():
                json_path = alt_path
            else:
                logger.warning(f"Enriched PTM data not found: {json_path}")
                return []

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} enriched PTMs from {json_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load enriched PTM data: {e}")
            return []

    def load_comprehensive_report(self, order_code: str, ptm_type: str = "phosphorylation") -> str:
        """Load the comprehensive markdown report."""
        suffix = "_phospho" if ptm_type == "phosphorylation" else "_ubi"
        md_path = self.artifacts_dir / order_code / f"comprehensive_report{suffix}.md"

        if not md_path.exists():
            alt_path = self.artifacts_dir / order_code / "comprehensive_report.md"
            if alt_path.exists():
                md_path = alt_path
            else:
                return ""

        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to load report: {e}")
            return ""

    def load_kinase_modules(self, order_code: str) -> Dict[str, Any]:
        """Load cached kinase module analysis."""
        path = self.artifacts_dir / order_code / "kinase_modules.json"
        return self._load_json(path)

    def load_signal_flow(self, order_code: str) -> Dict[str, Any]:
        """Load cached signal flow data."""
        path = self.artifacts_dir / order_code / "signal_flow.json"
        return self._load_json(path)

    def load_comovement_clusters(self, order_code: str) -> Dict[str, Any]:
        """Load temporal co-movement cluster data."""
        path = self.artifacts_dir / order_code / "comovement_clusters.json"
        return self._load_json(path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Generic JSON loader."""
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    # ─── DB-based reading (optional, for richer context) ─────────────────

    async def load_order_context(self, order_id: int) -> Dict[str, Any]:
        """Load order metadata from PTM-platform's MySQL (read-only)."""
        if not self.database_url:
            return {}

        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(self.database_url, pool_pre_ping=True)
            async with engine.connect() as conn:
                result = await conn.execute(
                    text("""
                        SELECT order_code, ptm_type, analysis_context,
                               report_options, kinase_analysis, receptor_inference,
                               signal_propagation
                        FROM orders WHERE id = :order_id
                    """),
                    {"order_id": order_id},
                )
                row = result.fetchone()
                if not row:
                    return {}

                return {
                    "order_code": row[0],
                    "ptm_type": row[1],
                    "analysis_context": json.loads(row[2]) if row[2] else {},
                    "report_options": json.loads(row[3]) if row[3] else {},
                    "kinase_analysis": json.loads(row[4]) if row[4] else {},
                    "receptor_inference": json.loads(row[5]) if row[5] else {},
                    "signal_propagation": json.loads(row[6]) if row[6] else {},
                }
        except Exception as e:
            logger.error(f"Failed to load order context from DB: {e}")
            return {}

    # ─── Context assembly (combines all sources) ─────────────────────────

    def assemble_context(self, order_code: str, ptm_type: str = "phosphorylation") -> Dict[str, Any]:
        """
        Assemble full context from PTM-platform artifacts for Co-Scientist.

        Returns a dict with all available data for hypothesis generation.
        """
        enriched = self.load_enriched_ptm_data(order_code, ptm_type)
        report = self.load_comprehensive_report(order_code, ptm_type)
        kinase = self.load_kinase_modules(order_code)
        signal = self.load_signal_flow(order_code)
        clusters = self.load_comovement_clusters(order_code)

        # Extract key PTM summaries
        top_ptms = []
        for ptm in enriched[:20]:
            top_ptms.append({
                "gene": ptm.get("gene", ptm.get("Gene.Name", "")),
                "position": ptm.get("position", ptm.get("Position", "")),
                "ptm_type": ptm.get("ptm_type", ptm_type),
                "ptm_relative_log2fc": ptm.get("ptm_relative_log2fc", 0),
                "protein_log2fc": ptm.get("protein_log2fc", 0),
                "pathways": ptm.get("rag_enrichment", {}).get("pathways", [])[:3],
                "function_summary": ptm.get("rag_enrichment", {}).get("function_summary", ""),
                "regulation": ptm.get("rag_enrichment", {}).get("regulation", {}),
            })

        return {
            "order_code": order_code,
            "ptm_type": ptm_type,
            "enriched_ptm_count": len(enriched),
            "top_ptms": top_ptms,
            "comprehensive_report_excerpt": report[:5000] if report else "",
            "kinase_modules": kinase,
            "signal_flow": signal,
            "comovement_clusters": clusters,
        }

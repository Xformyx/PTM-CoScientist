"""
PTM-Platform Connector — Read-only access to PTM-platform's artifacts.

Reads enriched PTM data (JSON files), order metadata (MySQL),
and cached analysis results (kinase modules, signal flow, etc.)
without modifying any PTM-platform data.
"""

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse numeric PTM values without failing the whole context assembly."""
    if value in (None, "", "NA", "NaN", "nan", "-", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_module_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        inner = raw.get("kinase_modules", [])
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def _run_async(coro):
    """Run an async coroutine from sync pipeline code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - surface connector failures to the caller
            error["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if "error" in error:
        raise error["error"]
    return result.get("value", {})


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
            if isinstance(data, dict):
                data = data.get("ptms") or data.get("data") or data.get("enriched_ptms") or []
            if not isinstance(data, list):
                logger.warning(f"Unexpected enriched PTM payload type in {json_path}: {type(data)}")
                return []
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

    def load_order_db_context(self, order_code: str) -> Dict[str, Any]:
        """Load kinase / receptor / signal-flow JSON stored on the Order row."""
        if not self.database_url or not order_code:
            return {}
        try:
            return _run_async(self._load_order_db_async(order_code=order_code, order_id=None))
        except Exception as e:
            logger.error(f"Failed to load order context from DB for {order_code}: {e}")
            return {}

    async def load_order_context(self, order_id: int) -> Dict[str, Any]:
        """Load order metadata from PTM-platform's MySQL (read-only)."""
        if not self.database_url:
            return {}
        try:
            return await self._load_order_db_async(order_code="", order_id=order_id)
        except Exception as e:
            logger.error(f"Failed to load order context from DB: {e}")
            return {}

    async def _load_order_db_async(
        self,
        order_code: str = "",
        order_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(self.database_url, pool_pre_ping=True)
        try:
            async with engine.connect() as conn:
                if order_id is not None:
                    result = await conn.execute(
                        text("""
                            SELECT order_code, ptm_type, analysis_context,
                                   kinase_analysis_data, receptor_inference_data,
                                   signal_propagation_data, kinase_activity_heatmap
                            FROM orders WHERE id = :order_id
                        """),
                        {"order_id": order_id},
                    )
                else:
                    result = await conn.execute(
                        text("""
                            SELECT order_code, ptm_type, analysis_context,
                                   kinase_analysis_data, receptor_inference_data,
                                   signal_propagation_data, kinase_activity_heatmap
                            FROM orders WHERE order_code = :order_code
                        """),
                        {"order_code": order_code},
                    )
                row = result.fetchone()
                if not row:
                    return {}
                return self._normalize_db_row(row)
        finally:
            await engine.dispose()

    @staticmethod
    def _normalize_db_row(row) -> Dict[str, Any]:
        def _json(value: Any) -> Any:
            if value is None:
                return {}
            if isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {}
            return {}

        kinase = _json(row[3])
        receptors = _json(row[4])
        signal = _json(row[5])
        heatmap = _json(row[6])
        analysis_context = _json(row[2])
        modules = _as_module_list(kinase)
        return {
            "order_code": row[0],
            "ptm_type": row[1],
            "analysis_context": analysis_context,
            "kinase_modules": {"kinase_modules": modules} if modules else {},
            "kinase_analysis": kinase,
            "receptor_inference": receptors,
            "signal_flow": signal if isinstance(signal, dict) else {},
            "kinase_activity_heatmap": heatmap if isinstance(heatmap, dict) else {},
        }

    # ─── Context assembly (combines all sources) ─────────────────────────

    def assemble_multi_context(
        self,
        order_codes: List[str],
        ptm_type: str = "phosphorylation",
    ) -> Dict[str, Any]:
        """
        Assemble and synthesise context from multiple orders.

        Merges PTM data across experiments, annotates each site with how many
        orders it appears in, and surfaces cross-order kinase patterns.
        """
        if not order_codes:
            return {}
        if len(order_codes) == 1:
            return self.assemble_context(order_codes[0], ptm_type)

        all_ptm_maps: Dict[str, Dict[str, Any]] = {}  # key: "GENE-POS"
        all_kinase_modules: List[Dict[str, Any]] = []
        report_excerpts: List[str] = []
        total_enriched = 0

        for oc in order_codes:
            enriched = self.load_enriched_ptm_data(oc, ptm_type)
            total_enriched += len(enriched)

            for ptm in enriched[:30]:
                if not isinstance(ptm, dict):
                    continue
                gene = ptm.get("gene", ptm.get("Gene.Name", ""))
                pos = ptm.get("position", ptm.get("Position", ""))
                key = f"{gene}-{pos}"
                fc = _safe_float(ptm.get("ptm_relative_log2fc", 0))

                if key not in all_ptm_maps:
                    all_ptm_maps[key] = {
                        "gene": gene,
                        "position": pos,
                        "ptm_type": ptm.get("ptm_type", ptm_type),
                        "ptm_relative_log2fc": fc,
                        "protein_log2fc": _safe_float(ptm.get("protein_log2fc", 0)),
                        "pathways": (ptm.get("rag_enrichment") or {}).get("pathways", [])[:3],
                        "function_summary": (ptm.get("rag_enrichment") or {}).get("function_summary", ""),
                        "regulation": (ptm.get("rag_enrichment") or {}).get("regulation", {}),
                        "order_codes": [oc],
                        "occurrence_count": 1,
                    }
                else:
                    existing = all_ptm_maps[key]
                    existing["occurrence_count"] += 1
                    existing["order_codes"].append(oc)
                    # Keep the highest absolute log2fc
                    if abs(fc) > abs(existing["ptm_relative_log2fc"]):
                        existing["ptm_relative_log2fc"] = fc

            kinase = self.load_kinase_modules(oc) or self.load_order_db_context(oc).get("kinase_modules", {})
            for mod in _as_module_list(kinase):
                all_kinase_modules.append({**mod, "order_code": oc})

            report = self.load_comprehensive_report(oc, ptm_type)
            if report:
                report_excerpts.append(f"### Order: {oc}\n{report[:1500]}")

        # Sort: first by occurrence (cross-order sites first), then by |log2fc|
        top_ptms = sorted(
            all_ptm_maps.values(),
            key=lambda x: (-x["occurrence_count"], -abs(x["ptm_relative_log2fc"])),
        )[:25]

        cross_order_sites = [p for p in top_ptms if p["occurrence_count"] > 1]

        return {
            "order_codes": order_codes,
            "order_count": len(order_codes),
            "ptm_type": ptm_type,
            "enriched_ptm_count": total_enriched,
            "top_ptms": top_ptms,
            "cross_order_sites": cross_order_sites,
            "cross_order_site_count": len(cross_order_sites),
            "kinase_modules": {"kinase_modules": all_kinase_modules[:20]},
            "signal_flow": {},
            "comovement_clusters": {},
            "comprehensive_report_excerpt": "\n\n".join(report_excerpts)[:6000],
        }

    def assemble_context(self, order_code: str, ptm_type: str = "phosphorylation") -> Dict[str, Any]:
        """
        Assemble full context from PTM-platform artifacts for Co-Scientist.

        Returns a dict with all available data for hypothesis generation.
        """
        enriched = self.load_enriched_ptm_data(order_code, ptm_type)
        report = self.load_comprehensive_report(order_code, ptm_type)
        db_ctx = self.load_order_db_context(order_code)
        kinase = self.load_kinase_modules(order_code) or db_ctx.get("kinase_modules", {})
        signal = self.load_signal_flow(order_code) or db_ctx.get("signal_flow", {})
        clusters = self.load_comovement_clusters(order_code)

        # Extract key PTM summaries
        top_ptms = []
        for ptm in enriched[:20]:
            if not isinstance(ptm, dict):
                continue
            top_ptms.append({
                "gene": ptm.get("gene", ptm.get("Gene.Name", "")),
                "position": ptm.get("position", ptm.get("Position", "")),
                "ptm_type": ptm.get("ptm_type", ptm_type),
                "ptm_relative_log2fc": _safe_float(ptm.get("ptm_relative_log2fc", 0)),
                "protein_log2fc": _safe_float(ptm.get("protein_log2fc", 0)),
                "pathways": (ptm.get("rag_enrichment") or {}).get("pathways", [])[:3],
                "function_summary": (ptm.get("rag_enrichment") or {}).get("function_summary", ""),
                "regulation": (ptm.get("rag_enrichment") or {}).get("regulation", {}),
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
            "receptor_inference": db_ctx.get("receptor_inference", {}),
            "kinase_activity_heatmap": db_ctx.get("kinase_activity_heatmap", {}),
        }

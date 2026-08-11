"""
ChromaDB Connector — Read-only access to PTM-platform's RAG collections.

Queries existing ChromaDB collections (articles indexed by PTM-platform)
to retrieve literature evidence for hypothesis generation and validation.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class ChromaDBConnector:
    """Read-only connector to PTM-platform's ChromaDB instance."""

    def __init__(self, chromadb_url: str = "http://localhost:8000"):
        self._url = chromadb_url
        self._client: chromadb.HttpClient | None = None

    @property
    def client(self) -> chromadb.HttpClient:
        if self._client is None:
            host, port = self._parse_url(self._url)
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @staticmethod
    def _parse_url(url: str) -> tuple:
        """Parse URL into host and port."""
        url = url.replace("http://", "").replace("https://", "")
        parts = url.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8000
        return host, port

    def is_available(self) -> bool:
        """Check if ChromaDB is reachable."""
        try:
            self.client.heartbeat()
            return True
        except Exception as e:  # noqa: BLE001 - connector availability failures are expected at runtime
            logger.warning(f"ChromaDB not available: {e}")
            return False

    def list_collections(self) -> list[str]:
        """List all available collection names."""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:  # noqa: BLE001 - a failed remote listing should degrade gracefully
            logger.error(f"Failed to list collections: {e}")
            return []

    def search(
        self,
        query: str,
        collection_names: list[str] | None = None,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search selected collections and return ranked, citation-preserving records."""
        if not self.is_available():
            return []

        available = self.list_collections()
        targets = collection_names or available
        targets = [c for c in targets if c in available]

        if not targets:
            logger.warning("No valid collections to search")
            return []

        all_results = []
        for coll_name in targets:
            try:
                collection = self.client.get_collection(coll_name)
                results = collection.query(
                    query_texts=[query],
                    n_results=min(n_results, 20),
                    include=["documents", "metadatas", "distances"],
                )

                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                ids = results.get("ids", [[]])[0]

                for index, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                    all_results.append({
                        "id": ids[index] if index < len(ids) else "",
                        "document": doc,
                        "metadata": meta or {},
                        "distance": dist,
                        "collection": coll_name,
                        "source_type": self._infer_source_type(meta),
                    })
            except Exception as e:  # noqa: BLE001 - one bad collection must not prevent other read-only searches
                logger.error(f"Error querying collection {coll_name}: {e}")

        # ChromaDB cosine/L2 distance is lower for closer matches.
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]

    def search_for_hypothesis(
        self,
        hypothesis,
        collection_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Targeted search for hypothesis validation (mirrors PTM-platform's RAGRetriever)."""
        queries = [
            getattr(hypothesis, "condition", ""),
            getattr(hypothesis, "prediction", ""),
            getattr(hypothesis, "mechanism", ""),
        ]
        query_text = " ".join(q for q in queries if q)
        if not query_text.strip():
            return []
        return self.search(query_text, collection_names=collection_names, n_results=8)

    def search_for_ptm(
        self,
        gene: str,
        position: str,
        ptm_type: str = "phosphorylation",
        collection_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for literature about a specific PTM site."""
        query = f"{gene} {position} {ptm_type} function signaling"
        return self.search(query, collection_names=collection_names, n_results=5)

    def search_for_context(
        self,
        genes: list[str],
        ptm_type: str = "phosphorylation",
        collection_names: list[str] | None = None,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve broad PTM/pathway context for hypothesis generation."""
        if not genes:
            return []
        gene_str = " ".join(genes[:8])
        query = f"{gene_str} {ptm_type} signaling kinase pathway regulation"
        return self.search(query, collection_names=collection_names, n_results=n_results)

    @staticmethod
    def _infer_source_type(metadata: dict | None) -> str:
        """Infer source type from metadata (textbook, review, research)."""
        if not metadata:
            return "unknown"
        title = (metadata.get("title") or "").lower()
        source = (metadata.get("source") or "").lower()
        if any(kw in title for kw in ["textbook", "chapter", "handbook"]):
            return "textbook"
        if any(kw in title for kw in ["review", "survey", "overview"]):
            return "review"
        if "pubmed" in source or "pmc" in source:
            return "research_article"
        return "unknown"

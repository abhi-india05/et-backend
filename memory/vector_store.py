"""
Vector memory store with FAISS backend and per-user namespace isolation.

Supports:
- Per-user namespace to prevent cross-user contamination
- TTL-based document pruning
- Max document limits per user and globally
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config.settings import settings
from backend.llm.gemini_client import get_gemini_embedding
from backend.utils.logger import get_logger

logger = get_logger("vector_store")


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][vector_store][{level.upper()}] {message}")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

OPENAI_AVAILABLE = True


class VectorMemoryStore:
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.index_path = index_path or settings.faiss_index_path
        self.documents: List[Dict[str, Any]] = []
        self.index = None
        self._embedding_fn = None
        self._initialize()

    def _initialize(self) -> None:
        if not FAISS_AVAILABLE:
            logger.warning("faiss_unavailable", reason="faiss-cpu not installed, using in-memory fallback")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("faiss_index_initialized", dimension=self.dimension)

        if settings.has_gemini_key:
            try:
                self._embedding_fn = get_gemini_embedding
                logger.info("embedding_client_ready")
            except Exception as e:
                logger.warning("embedding_client_failed", error=str(e))
                self._embedding_fn = None

    def _get_embedding(self, text: str) -> np.ndarray:
        if not self._embedding_fn:
            raise RuntimeError(
                "Embedding client is not configured. "
                "Verify GEMINI_API_KEY before using vector memory."
            )

        try:
            embedding_list = self._embedding_fn(text)
            embedding = np.array(embedding_list, dtype=np.float32)
            preview = embedding[:8].tolist()
            _terminal_log("success", f"Embedding raw output (first 8 values): {preview}")
            logger.info("embedding_llm_output", output_preview=preview, dimension=len(embedding))
            return embedding
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            _terminal_log("failure", f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    # ------------------------------------------------------------------
    # Namespace-aware document operations
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        content: str,
        namespace: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a document with required namespace isolation.

        Args:
            doc_id: Unique identifier for the document.
            content: Text content to embed and store.
            namespace: User-level namespace for isolation (mandatory user_id).
            metadata: Arbitrary metadata dict (company, agent, session, etc.).
        """
        if not namespace or namespace == "global":
            raise ValueError("namespace (user_id) is strictly required for vector store operations.")
        try:
            # Enforce global document limit
            if len(self.documents) >= settings.memory_max_total_documents:
                self._prune_oldest(count=max(1, settings.memory_max_total_documents // 10))

            # Enforce per-user limit if namespace provided
            if namespace:
                user_docs = [d for d in self.documents if d.get("namespace") == namespace]
                if len(user_docs) >= settings.memory_max_documents_per_user:
                    self._prune_oldest_for_namespace(
                        namespace,
                        count=max(1, settings.memory_max_documents_per_user // 10),
                    )

            embedding = self._get_embedding(content)
            now = datetime.now(timezone.utc)
            doc: Dict[str, Any] = {
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "namespace": namespace,
                "timestamp": now.isoformat(),
                "created_at_epoch": now.timestamp(),
                "embedding_index": len(self.documents),
            }
            self.documents.append(doc)

            if self.index is not None and FAISS_AVAILABLE:
                self.index.add(embedding.reshape(1, -1))

            return True
        except Exception as e:
            logger.error("add_document_failed", doc_id=doc_id, error=str(e))
            return False

    def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents, scoped to a namespace."""
        if not namespace or namespace == "global":
            raise ValueError("namespace (user_id) is strictly required for search.")
            
        if not self.documents:
            return []

        try:
            query_embedding = self._get_embedding(query)

            if self.index is not None and FAISS_AVAILABLE and self.index.ntotal > 0:
                # Search more candidates than top_k so we can filter by namespace
                search_k = min(top_k * 5, len(self.documents))
                distances, indices = self.index.search(query_embedding.reshape(1, -1), search_k)
                candidates = []
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self.documents):
                        doc = dict(self.documents[idx])
                        doc["similarity_score"] = float(1.0 / (1.0 + dist))
                        candidates.append(doc)
            else:
                candidates = [dict(d) for d in self.documents[-top_k * 3 :]]
                for r in candidates:
                    r["similarity_score"] = 0.5

            # Filter by namespace
            candidates = [r for r in candidates if r.get("namespace") == namespace]

            # Filter by metadata
            if filter_metadata:
                candidates = [
                    r
                    for r in candidates
                    if all(r.get("metadata", {}).get(k) == v for k, v in filter_metadata.items())
                ]

            return candidates[:top_k]

        except Exception as e:
            logger.error("search_failed", error=str(e))
            return []

    def get_context_for_company(
        self,
        company: str,
        namespace: str,
    ) -> str:
        if not namespace or namespace == "global":
            raise ValueError("namespace (user_id) is required.")
            
        results = self.search(company, namespace=namespace, top_k=3, filter_metadata={"company": company})
        if not results:
            results = self.search(company, namespace=namespace, top_k=3)

        if not results:
            return "No prior context available."

        return "\n".join(f"[{r['timestamp'][:10]}] {r['content']}" for r in results)

    # ------------------------------------------------------------------
    # TTL & pruning
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Remove documents older than the configured TTL. Returns count removed."""
        if not self.documents:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - settings.memory_ttl_seconds
        before = len(self.documents)
        self.documents = [d for d in self.documents if d.get("created_at_epoch", 0) > cutoff]
        removed = before - len(self.documents)
        if removed > 0:
            self._rebuild_index()
            logger.info("memory_pruned_ttl", removed=removed)
        return removed

    def _prune_oldest(self, count: int) -> None:
        """Remove the oldest `count` documents globally."""
        self.documents = self.documents[count:]
        self._rebuild_index()
        logger.info("memory_pruned_global", removed=count)

    def _prune_oldest_for_namespace(self, namespace: str, count: int) -> None:
        """Remove the oldest `count` documents for a specific namespace."""
        removed = 0
        new_docs: List[Dict[str, Any]] = []
        for doc in self.documents:
            if doc.get("namespace") == namespace and removed < count:
                removed += 1
                continue
            new_docs.append(doc)
        self.documents = new_docs
        self._rebuild_index()
        logger.info("memory_pruned_namespace", namespace=namespace, removed=removed)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current documents."""
        if not FAISS_AVAILABLE:
            return
        self.index = faiss.IndexFlatL2(self.dimension)
        for doc in self.documents:
            embedding = self._get_embedding(doc["content"])
            self.index.add(embedding.reshape(1, -1))
            doc["embedding_index"] = self.index.ntotal - 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        try:
            index_dir = os.path.dirname(self.index_path)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)

            with open(f"{self.index_path}_docs.pkl", "wb") as f:
                pickle.dump(self.documents, f)

            if self.index is not None and FAISS_AVAILABLE:
                faiss.write_index(self.index, f"{self.index_path}.faiss")

            logger.info("vector_store_saved", doc_count=len(self.documents))
        except Exception as e:
            logger.error("vector_store_save_failed", error=str(e))

    def load(self) -> None:
        try:
            docs_path = f"{self.index_path}_docs.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                logger.info("vector_store_loaded", doc_count=len(self.documents))
            else:
                logger.info("vector_store_fresh", reason="no persisted index found")

            faiss_path = f"{self.index_path}.faiss"
            if FAISS_AVAILABLE and os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
                logger.info("faiss_index_loaded")

        except Exception as e:
            logger.error("vector_store_load_failed", error=str(e))
            self.documents = []
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.dimension)

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all documents, or only those in a specific namespace."""
        if namespace:
            self.documents = [d for d in self.documents if d.get("namespace") != namespace]
            self._rebuild_index()
            logger.info("vector_store_namespace_cleared", namespace=namespace)
        else:
            self.documents = []
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("vector_store_cleared")

    def stats(self) -> Dict[str, Any]:
        namespaces: Dict[str, int] = {}
        for doc in self.documents:
            ns = doc.get("namespace", "global")
            namespaces[ns] = namespaces.get(ns, 0) + 1
        return {
            "total_documents": len(self.documents),
            "faiss_available": FAISS_AVAILABLE,
            "dimension": self.dimension,
            "index_active": self.index is not None,
            "embedding_client": self._embedding_fn is not None,
            "namespaces": namespaces,
        }


_store_instance: Optional[VectorMemoryStore] = None


def get_vector_store() -> VectorMemoryStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorMemoryStore()
    return _store_instance

from typing import List, Dict, Optional, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorSearchProvider(ABC):
    @abstractmethod
    def add_products(self, products: List[Dict[str, Any]]) -> None: ...

    @abstractmethod
    def add_troubleshooting(self, entries: List[Dict[str, Any]]) -> None: ...

    @abstractmethod
    def search_products(self, query: str, top_k: int = 5, **filters) -> List[Dict]: ...

    @abstractmethod
    def search_troubleshooting(self, query: str, top_k: int = 3, **filters) -> List[Dict]: ...

    @abstractmethod
    def get_product_by_part_number(self, part_number: str) -> Optional[Dict]: ...


class ChromaVectorDB(VectorSearchProvider):

    def __init__(self):
        self.enabled = False
        self.products_col = None
        self.troubleshoot_col = None
        self.embedder = None

        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.chroma = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
            self.products_col = self.chroma.get_or_create_collection("products")
            self.troubleshoot_col = self.chroma.get_or_create_collection("troubleshooting")
            self.enabled = True
            logger.info("VectorDB initialized with Chroma + MiniLM.")
        except Exception as e:
            logger.warning(f"VectorDB unavailable, continuing without vector search ({e})")
            self.enabled = False

    # --- ingestion ---
    def add_products(self, products: List[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        # If already populated, skip re-upsert to avoid duplicate IDs
        try:
            if self.products_col.count() > 0:
                return
        except Exception:
            pass
        # Otherwise, reset and insert
        self.products_col = self._reset_collection("products")
        ids, embeddings, metas = [], [], []
        seen = set()

        for p in products:
            pid = p.get("part_number")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            # store only primitive metadata for Chroma
            metas.append({
                "part_number": pid,
                "name": p.get("name"),
                "price": p.get("price"),
                "in_stock": p.get("in_stock"),
                "availability": p.get("availability"),
                "appliance_type": p.get("appliance_type"),
                "brand": p.get("brand"),
                "category": p.get("category"),
                "compatible_models": ",".join(p.get("compatible_models", [])),
                "description": p.get("description"),
                "product_url": p.get("product_url"),
                "main_image": p.get("main_image"),
                "manufacturer": p.get("manufacturer"),
                "manufacturer_part_number": p.get("manufacturer_part_number"),
                "installation_time": p.get("installation_time"),
                "installation_complexity": p.get("installation_complexity"),
                "rating_value": p.get("rating_value"),
                "rating_count": p.get("rating_count"),
                "replaces": ",".join(p.get("replaces", [])),
                "symptoms": ",".join(p.get("symptoms", [])),
            })
            embeddings.append(self._embed(f"{p['name']} {p['description']} {p['brand']} {p['category']} {p['appliance_type']}"))
        if ids:
            self.products_col.upsert(ids=ids, embeddings=embeddings, metadatas=metas)

    def add_troubleshooting(self, entries: List[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        try:
            if self.troubleshoot_col.count() > 0:
                return
        except Exception:
            pass
        self.troubleshoot_col = self._reset_collection("troubleshooting")
        ids, embeddings, metas = [], [], []
        for idx, t in enumerate(entries):
            ids.append(str(idx))
            symptom = t.get("symptom") or t.get("symptom_slug") or t.get("symptom_display") or ""
            embeddings.append(self._embed(f"{t.get('appliance_type','')} {symptom} {' '.join(t.get('common_causes', []))}"))
            metas.append({
                "appliance_type": t.get("appliance_type"),
                "symptom": symptom,
                "common_causes": "; ".join(t.get("common_causes", [])),
            })
        if ids:
            self.troubleshoot_col.upsert(ids=ids, embeddings=embeddings, metadatas=metas)

    # --- queries ---
    def search_products(
        self,
        query: str,
        top_k: int = 5,
        **filters,
    ) -> List[Dict]:
        if not self.enabled:
            return []
        embedding = self._embed(query)
        fltr = {}
        appliance_type = filters.get("appliance_type")
        brand = filters.get("brand")
        category = filters.get("category")
        if appliance_type:
            fltr["appliance_type"] = appliance_type
        if brand:
            fltr["brand"] = brand
        if category:
            fltr["category"] = category

        res = self.products_col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=fltr or None,
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        out = []
        for i, pid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            meta = {**meta, "relevance_score": 1 - res["distances"][0][i]}
            out.append(meta)
        return out

    def get_product_by_part_number(self, part_number: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        try:
            res = self.products_col.get(where={"part_number": part_number})
            if res and res.get("metadatas"):
                return res["metadatas"][0]
        except Exception:
            return None
        return None

    def search_troubleshooting(
        self,
        query: str,
        top_k: int = 3,
        **filters,
    ) -> List[Dict]:
        if not self.enabled:
            return []
        embedding = self._embed(query)
        appliance_type = filters.get("appliance_type")
        fltr = {"appliance_type": appliance_type} if appliance_type else None
        res = self.troubleshoot_col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=fltr,
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        out = []
        for i, _id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            meta = {**meta, "relevance_score": 1 - res["distances"][0][i]}
            out.append(meta)
        return out

    # --- helpers ---
    def _embed(self, text: str):
        return self.embedder.encode(text).tolist()

    def _reset_collection(self, name: str):
        try:
            if name in [c.name for c in self.chroma.list_collections()]:
                self.chroma.delete_collection(name)
        except Exception:
            pass
        return self.chroma.get_or_create_collection(name)

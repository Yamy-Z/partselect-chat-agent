from typing import Dict, List, Any, Optional
import re
import logging
import asyncio

from ..cache import cache
from ..vector_db import VectorSearchProvider
from ..llm import get_llm

logger = logging.getLogger(__name__)


class ProductAgent:

    def __init__(self, products_data: Optional[List[Dict]] = None, vector_db: Optional[VectorSearchProvider] = None):
        self.products = products_data or []
        self.vector_db = vector_db
        self.llm = get_llm()
        # Keep a lightweight lookup for enrichment (part_number -> full record)
        self.by_part = {p["part_number"]: p for p in self.products}

    async def search(self, entities: Dict, user_message: str) -> Dict[str, Any]:
        """
        Return relevant products given extracted entities or free text.
        """
        cache_key = f"prod:{entities}:{user_message}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Exact part number (prefer vector DB to avoid holding huge indexes)
        part_number = entities.get("part_number")
        if part_number:
            product = None
            if self.vector_db and self.vector_db.enabled:
                product = self._enrich_product(self.vector_db.get_product_by_part_number(part_number))
            if not product and part_number in self.by_part:
                product = self.by_part[part_number]
            if product:
                response = {
                    "response": self._format_single(product),
                    "products": [product],
                }
                cache.set(cache_key, response, ttl=900)
                return response

        # Semantic search if available
        if self.vector_db and self.vector_db.enabled:
            candidates = self._vector_db_search(
                user_message=user_message,
                appliance_type=entities.get("appliance_type"),
                brand=entities.get("brand"),
                part_type=entities.get("part_type"),
                model_number=entities.get("model_number"),
            )
            candidates = self._enrich_products(candidates)
        else:
            # run keyword scoring off-thread to avoid blocking event loop
            candidates = await asyncio.to_thread(
                self._score_products,
                user_message,
                entities.get("appliance_type"),
                entities.get("brand"),
                entities.get("part_type"),
                entities.get("model_number"),
            )

        if not candidates:
            response = {
                "response": f"I couldn't find parts that match '{user_message}'. Tell me the appliance type, brand, and any part number you have. I can help with refrigerators and dishwashers.",
                "products": [],
            }
            cache.set(cache_key, response, ttl=300)
            return response

        llm_text = await self._generate_response(candidates, entities, user_message)
        response = {
            "response": llm_text,
            "products": candidates,
            "steps": self._installation_steps(candidates, user_message),
        }
        cache.set(cache_key, response, ttl=900)
        return response

    async def get_info(self, entities: Dict, user_message: str) -> Dict[str, Any]:
        """
        If part number present return details; else fall back to search.
        """
        part_number = entities.get("part_number")
        model_number = entities.get("model_number")
        compatibility_only = any(
            kw in user_message.lower() for kw in ["compatib", "fit", "work with", "works with", "fit my", "compatible with"]
        )
        if part_number:
            product = None
            if self.vector_db and self.vector_db.enabled:
                product = self._enrich_product(self.vector_db.get_product_by_part_number(part_number))
            if not product and part_number in self.by_part:
                product = self.by_part[part_number]
            if not product:
                return await self.search(entities, user_message)

            # Compatibility-only flow: return a concise yes/no without cards/steps
            if compatibility_only and model_number:
                is_compatible = model_number in product.get("compatible_models", [])
                verdict = "Yes" if is_compatible else "No"
                resp = f"{verdict}. {product['name']} ({product['part_number']}) " \
                       f"{'fits' if is_compatible else 'is not listed as compatible with'} model {model_number}."
                if not is_compatible:
                    resp += " I can help you find the right part for that model."
                return {"response": resp, "products": [], "steps": []}

            llm_text = await self._generate_response([product], entities, user_message, detailed=True)
            return {
                "response": llm_text,
                "products": [product],
                "steps": self._installation_steps([product], user_message),
            }
        return await self.search(entities, user_message)

    # Helpers
    def _score_products(
        self,
        user_message: str,
        appliance_type: Optional[str],
        brand: Optional[str],
        part_type: Optional[str],
        model_number: Optional[str],
    ) -> List[Dict]:
        """Simple keyword scoring to keep dependencies light."""
        terms = self._tokenize(user_message)
        scored = []
        pool = self.products
        for p in pool:
            score = 0
            if appliance_type and p["appliance_type"] == appliance_type:
                score += 3
            if brand and p["brand"].lower() == brand.lower():
                score += 2
            if part_type and part_type.lower() in p["category"].lower():
                score += 2
            if model_number and model_number in p.get("compatible_models", []):
                score += 3
            # keyword hits
            for term in terms:
                if term in p["name"].lower() or term in p["description"].lower():
                    score += 1
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:5]]

    def _vector_db_search(
        self,
        user_message: str,
        appliance_type: Optional[str],
        brand: Optional[str],
        part_type: Optional[str],
        model_number: Optional[str],
        top_k: int = 5,
    ) -> List[Dict]:
        if not self.vector_db or not self.vector_db.enabled:
            return []

        query = user_message
        if appliance_type:
            query += f" {appliance_type}"
        if brand:
            query += f" {brand}"
        if part_type:
            query += f" {part_type}"

        hits = self.vector_db.search_products(
            query=query,
            appliance_type=appliance_type,
            brand=brand,
            category=part_type,
            top_k=top_k,
        )

        # If model filter provided, keep only compatible matches
        if model_number:
            hits = [h for h in hits if model_number in h.get("compatible_models", [])]

        return hits

    def _enrich_product(self, prod: Optional[Dict]) -> Optional[Dict]:
        if not prod:
            return None
        original = self.by_part.get(prod.get("part_number"))
        if not original:
            return prod
        merged = {**prod, **original}
        return merged

    def _enrich_products(self, products: List[Dict]) -> List[Dict]:
        out = []
        for p in products:
            enriched = self._enrich_product(p) or p
            out.append(enriched)
        return out

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _format_single(product: Dict, detailed: bool = False) -> str:
        base = (
            f"{product['name']} ({product['part_number']}) - ${product['price']:.2f}. "
            f"{'In stock' if product['in_stock'] else 'Currently out of stock'}."
        )
        if detailed:
            extra = (
                f" Type: {product['appliance_type']}, category: {product['category']}, brand: {product['brand']}. "
                f"Compatible models: {', '.join(product.get('compatible_models', [])[:5]) or '—'}. "
                f"Install time ~{product.get('installation_time_minutes', 'N/A')} min."
            )
            return base + " " + extra
        return base + " Ask if you want compatibility or installation steps."

    @staticmethod
    def _format_list(products: List[Dict]) -> str:
        lines = []
        for p in products:
            lines.append(
                f"- {p['name']} ({p['part_number']}) ${p['price']:.2f} | {p['appliance_type']} | "
                f"{'In stock' if p['in_stock'] else 'Out of stock'}"
            )
        return "Here are matches:\n" + "\n".join(lines)

    async def _generate_response(self, products: List[Dict], entities: Dict, user_message: str, detailed: bool = False) -> str:
        """
        Use OpenAI to compose a natural answer given retrieved products.
        """
        products_snippet = []
        for p in products[:3]:
            products_snippet.append(
                f"{p['name']} (#{p['part_number']}), ${p['price']}, "
                f"{'in stock' if p['in_stock'] else 'out of stock'}, "
                f"category: {p['category']}, appliance: {p['appliance_type']}, "
                f"compatible: {', '.join(p.get('compatible_models', [])[:5])}"
            )
        prompt = f"""User asked: {user_message}

Retrieved products:
- {"; ".join(products_snippet)}

Write a concise, helpful answer as a PartSelect assistant. Mention price and stock, offer compatibility help, and invite the user to ask for installation steps if relevant. Keep it to 3-4 sentences."""

        text = await self.llm.generate(
            prompt,
            temperature=0.3,
            max_tokens=220,
            response_format=None,
        )
        return text

    def _call_llm(self, prompt: str) -> str:
        return ""

    def _installation_steps(self, products: List[Dict], user_message: str) -> List[Dict]:
        """
        Return lightweight steps when the user hints at installation OR when we only
        surfaced a single product that already contains curated steps. This keeps the
        assistant helpful without overwhelming multi-result answers.
        """
        if not products:
            return []

        want_install = "install" in user_message.lower() or "replace" in user_message.lower()
        single_product = len(products) == 1

        if not (want_install or single_product):
            return []

        steps = products[0].get("installation_steps", []) or []
        return [{"step": idx + 1, "title": None, "detail": s} for idx, s in enumerate(steps)]

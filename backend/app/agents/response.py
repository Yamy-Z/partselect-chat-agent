import os
import asyncio
import json
from typing import Dict, Any, List
import google.generativeai as genai


class ResponseAgent:
    """
    Generates the final user-facing reply using LLM with retrieved context.
    Returns structured JSON we can pass directly to the frontend.
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    async def generate(self, user_message: str, intent: str, context: Dict[str, Any], history=None) -> Dict[str, Any]:
        """
        context: {"products": [...], "steps": [...]}
        Returns dict with keys: message, products (normalized), steps.
        """
        products: List[Dict] = context.get("products", []) or []
        steps: List[Dict] = context.get("steps", []) or []
        history = history or []

        prompt = self._build_prompt(user_message, intent, products, steps, history)
        raw = await asyncio.to_thread(self._call_llm, prompt)

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}

        # Fallback population
        return {
            "message": parsed.get("message") or raw.strip(),
            "products": parsed.get("products") or products,
            "steps": parsed.get("steps") or steps,
        }

    def _build_prompt(self, user_message: str, intent: str, products: List[Dict], steps: List[Dict], history) -> str:
        last_turns = history[-3:]
        product_cards = []
        for p in products[:5]:
            product_cards.append(
                {
                    "name": p.get("name"),
                    "part_number": p.get("part_number"),
                    "price": p.get("price"),
                    "in_stock": p.get("in_stock"),
                    "availability": p.get("availability"),
                    "appliance_type": p.get("appliance_type"),
                    "category": p.get("category"),
                    "compatible_models": p.get("compatible_models", [])[:5],
                    "installation_time_minutes": p.get("installation_time_minutes"),
                    "product_url": p.get("product_url") or p.get("documentation_url") or "",
                    "main_image": p.get("main_image") or p.get("image_url") or "",
                    "manufacturer": p.get("manufacturer") or p.get("brand"),
                    "manufacturer_part_number": p.get("manufacturer_part_number"),
                    "replaces": p.get("replaces", []),
                    "symptoms": p.get("symptoms", []),
                    "rating_value": p.get("rating_value"),
                    "rating_count": p.get("rating_count"),
                    "model_cross_reference": p.get("model_cross_reference", [])[:5],
                    "description": p.get("description"),
                }
            )

        step_items = [
            {
                "step": s.get("step"),
                "detail": s.get("detail") or s.get("title") or "",
            }
            for s in steps[:8]
        ]

        schema = {
            "message": "string",
            "products": [
                {
                    "name": "string",
                    "part_number": "string",
                    "price": "number",
                    "in_stock": "bool",
                    "availability": "string",
                    "appliance_type": "string",
                    "category": "string",
                    "compatible_models": ["string"],
                    "installation_time_minutes": "number",
                    "product_url": "string",
                    "main_image": "string",
                    "manufacturer": "string",
                    "manufacturer_part_number": "string",
                    "replaces": ["string"],
                    "symptoms": ["string"],
                    "rating_value": "number",
                    "rating_count": "number",
                    "model_cross_reference": [{"brand": "string", "model_number": "string", "model_url": "string", "description": "string"}],
                    "description": "string",
                }
            ],
            "steps": [
                {"step": "int", "detail": "string"}
            ],
        }

        return f"""You are PartSelect's chat agent.
User asked: {user_message}
Intent: {intent}
Recent history: {last_turns}
Products JSON: {product_cards}
Steps JSON: {step_items}

Produce STRICT JSON only (no markdown, no extra text) matching this schema:
{schema}

Rules:
- message: 3-5 sentences, answer the user's query directly, only choose necessary information from product, do not include non-relevant information from user's message.
- products: reuse provided products, do not invent new parts; keep provided URLs/images if present, otherwise use empty string.
- steps: reuse provided steps as-is.
- No markdown, only JSON."""

    def _call_llm(self, prompt: str) -> str:
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return resp.text or ""

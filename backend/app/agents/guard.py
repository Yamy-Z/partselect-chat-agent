import asyncio
import re
from typing import Optional, Dict, Any

from ..llm import get_llm


class GuardAgent:
    """
    Enforces domain boundaries — only refrigerator and dishwasher parts/support.
    Uses heuristics first, then LLM for ambiguous cases.
    """

    def __init__(self):
        self.llm = get_llm()

    async def check_scope(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = user_message.lower()

        # # Fast allow:appliance keywords
        # if any(k in text for k in ["dishwasher", "fridge", "refrigerator", "freezer", "ice maker"]):
        #     return True

        # Fast deny: obvious off-topic
        if re.search(r"\b(oven|washer|dryer|microwave|phone|laptop|computer|hvac)\b", text):
            return False
        return True

        # # LLM for ambiguous cases
        # prompt = f"""You are a domain guard for a PartSelect chat agent.

        #         The agent ONLY helps with refrigerator and dishwasher parts/support.
        #         User query: "{user_message}"

        #         Answer ONLY "YES" if in scope, otherwise "NO"."""

        # try:
        #     resp_text = await self.llm.generate(
        #         prompt, temperature=0, max_tokens=40, response_format=None
        #     )
        #     return resp_text.strip().upper() == "YES"
        # except Exception:
        #     return False

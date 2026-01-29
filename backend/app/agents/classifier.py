import json
import asyncio
from typing import Dict, List, Any

from ..llm import get_llm


class QueryClassifier:
    """
    Analyzes user queries to determine intent and extract entities
    """
    
    def __init__(self):
        self.llm = get_llm()
    
    async def classify(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Classify user intent and extract entities
        
        """
        
        # Build conversation context
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-5:]  # Last 5 messages
        ])
        
        prompt = f"""You are a query classification assistant for a PartSelect chat agent specializing in refrigerator and dishwasher parts.

                Conversation history:
                {context}

                Current user query: {user_message}

                Your task:
                1. Determine the user's intent from these categories:
                - compatibility_check: Verifying if a part fits a model
                - installation_help: How to install a part
                - troubleshooting: Diagnosing appliance issues
                - general_info: General questions about parts/models

                2. Extract relevant entities:
                - part_number (e.g., PS11752778)
                - model_number (e.g., WDT780SAEM1)
                - brand (e.g., Whirlpool, GE, Samsung)
                - appliance_type (refrigerator or dishwasher)
                - symptom (e.g., "ice maker not working", "leaking water")

                Return ONLY a JSON object with this exact format:
                {{
                    "intent": "intent_name",
                    "entities": {{
                        "part_number": "value or null",
                        "model_number": "value or null",
                        "brand": "value or null",
                        "appliance_type": "value or null",
                        "symptom": "value or null"
                    }}
                }}"""

        text = await self.llm.generate(
            prompt,
            temperature=0,
            max_tokens=500,
            response_format="json",
        )
        try:
            result = json.loads(text)
        except Exception as e:
            # Fallback to a safe default if the model returns malformed JSON
            result = {
                "intent": "general_info",
                "entities": {
                    "part_number": None,
                    "model_number": None,
                    "brand": None,
                    "appliance_type": None,
                    "symptom": None,
                },
            }
        return result

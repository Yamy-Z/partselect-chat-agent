import os
import asyncio
import logging
from typing import Optional
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as g_exceptions

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[str] = None,
    ) -> str:
        ...


class LLMQuotaError(Exception):
    """Raised when LLM provider returns a quota / rate limit error."""


class LLMProviderError(Exception):
    """Raised when LLM provider fails for other reasons."""


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self):
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[str] = None,
    ) -> str:
        return await self._generate_with_retry(prompt, temperature, max_tokens, response_format)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[str],
    ) -> str:
        import google.generativeai as genai

        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_format == "json":
            config["response_mime_type"] = "application/json"

        model = genai.GenerativeModel(self.model_name)
        try:
            response = await asyncio.to_thread(model.generate_content, prompt, generation_config=config)
            return response.text or ""
        except g_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini quota exceeded: {e}")
            raise LLMQuotaError(str(e)) from e
        except g_exceptions.GoogleAPICallError as e:
            logger.error(f"Gemini API error: {e}")
            raise LLMProviderError(str(e)) from e
        except Exception as e:
            logger.error(f"Gemini unknown error: {e}")
            raise LLMProviderError(str(e)) from e


class LLMFactory:
    @staticmethod
    def create() -> BaseLLMProvider:
        # Currently default to Gemini; extendable for other providers.
        return GeminiProvider()


_llm_instance: Optional[BaseLLMProvider] = None


def get_llm() -> BaseLLMProvider:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMFactory.create()
        logger.info(f"Initialized LLM provider: {_llm_instance.__class__.__name__}")
    return _llm_instance

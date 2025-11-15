from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

Message = Dict[str, str]


class BaseLLMClient(ABC):
    """Base interface for LLM clients."""

    @abstractmethod
    def get_llm_response(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Generate a response given chat messages."""

    @abstractmethod
    def _test_connection(self) -> bool:
        """Verify we can complete a minimal request."""


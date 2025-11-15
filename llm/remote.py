from __future__ import annotations

from typing import Any, Dict, List, Optional

import os

import dotenv
from huggingface_hub import InferenceClient

from .base import BaseLLMClient, Message


class RemoteLLMClient(BaseLLMClient):
    """
    Client for using remote LLM APIs via HuggingFace Inference.
    
    This client connects to HuggingFace-hosted or HuggingFace-compatible services.
    Requires an API key and internet connection.
    
    Example:
        >>> client = RemoteLLMClient(api_key="your_hf_token")
        >>> response = client.get_llm_response(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="openai/gpt-oss-20b:groq",
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize remote LLM client.
        
        Args:
            api_key: HuggingFace API key (optional, reads from HF_TOKEN env var if None)
            default_model: Optional default model to use for chat completions
        
        Raises:
            ValueError: If API key is not provided and not found in environment
            Exception: If client initialization fails
        """
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.default_model = default_model or os.getenv("HF_MODEL")

        if not self.api_key:
            raise ValueError(
                "HF_TOKEN not found.\n"
                "Please either:\n"
                "  1. Set HF_TOKEN in your .env file, or\n"
                "  2. Pass api_key parameter when creating RemoteLLMClient\n"
                "Get your API key at: https://huggingface.co/settings/tokens"
            )

        try:
            self.client = InferenceClient(api_key=self.api_key)
        except Exception as exc:
            raise ValueError(
                "Failed to initialize HuggingFace Inference Client.\n"
                f"Error: {exc}\n"
                "Please check that your HF_TOKEN is correct."
            ) from exc

        self._test_connection()

    def get_llm_response(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0,
        response_format: Optional[Dict[str, str]] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Get a response from the remote LLM API.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            model: Model name (e.g., "openai/gpt-oss-20b:groq")
            temperature: Sampling temperature (0 = deterministic)
            response_format: Optional dict with "type": "json_object" for structured output
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            The generated text response, or None if API call failed
        
        Raises:
            ValueError: If model parameter is not provided
        """
        target_model = model or self.default_model

        if target_model is None:
            raise ValueError(
                "model parameter is required for RemoteLLMClient.\n"
                "Example: model='openai/gpt-oss-20b:groq'"
            )

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            payload["response_format"] = response_format

        if max_new_tokens is not None:
            payload["max_tokens"] = max_new_tokens

        payload.update(kwargs)

        try:
            response = self.client.chat.completions.create(**payload)
            message = response.choices[0].message if response.choices else None
            if isinstance(message, dict):
                return message.get("content")
            return getattr(message, "content", None)
        except Exception as exc:
            error_msg = str(exc)
            if "rate limit" in error_msg.lower():
                print(
                    "Warning: API rate limit exceeded. Please wait a moment and try again.\n"
                    f"Error: {error_msg}"
                )
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                print(
                    "Warning: API authentication failed. Please check your HF_TOKEN.\n"
                    f"Error: {error_msg}"
                )
            else:
                print(
                    "Warning: Remote LLM API call failed.\n"
                    f"Error: {error_msg}\n"
                    "This request will be skipped."
                )
            return None

    def _test_connection(self) -> bool:
        """
        Test that the remote LLM API is working correctly.
        
        Raises:
            Exception: If the test fails (API not responding correctly)
        """
        test_model = self.default_model or os.getenv("HF_TEST_MODEL")

        if not test_model:
            print("Skipping remote LLM connection test (no default model configured)")
            return True

        try:
            response = self.client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "Please confirm this test."}],
                temperature=0,
                max_tokens=16,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "authentication" in error_msg or "token" in error_msg:
                raise ValueError(
                    "Failed to connect to remote LLM - authentication error.\n"
                    f"Error: {exc}\n"
                    "Please check your HF_TOKEN is correct and valid."
                ) from exc
            else:
                raise RuntimeError(
                    "Failed to connect to remote LLM.\n"
                    f"Error: {exc}\n"
                    "Please check your API key, default model (if provided), and internet connection."
                ) from exc

        message = response.choices[0].message if response.choices else None
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

        if content:
            print(f"✓ Remote LLM connection test successful ({test_model})")
            return True

        raise RuntimeError(
            "Remote LLM test response was empty.\n"
            "The API connection works, but the model did not return a response."
        )


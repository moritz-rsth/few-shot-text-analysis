from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import login as hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLMClient, Message
from .utils import extract_json_block


class LocalLLMClient(BaseLLMClient):
    """
    Client for running LLM models locally using HuggingFace transformers.
    
    This client loads models from HuggingFace and runs them on your local machine.
    Supports models like Llama, Qwen, and other HuggingFace-compatible models.
    
    Example:
        >>> client = LocalLLMClient(
        ...     base_url="meta-llama/Llama-3.1-8B-Instruct",
        ...     hf_token="your_token_here",  # Optional
        ... )
        >>> response = client.get_llm_response(messages=[...])
    """

    def __init__(
        self,
        base_url: str,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize local LLM client.
        
        Args:
            base_url: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            hf_token: HuggingFace token (optional, reads from HF_TOKEN env var if None)
        
        Note:
            The model will be downloaded on first use if not already cached.
            This may take several minutes depending on model size and internet speed.
        """
        self.model_id = base_url
        self.model = None
        self.tokenizer = None

        self._login(hf_token)
        self._load_model()
        self._test_connection()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _login(self, hf_token: Optional[str]) -> None:
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            return
        try:
            hf_login(token=token)
        except Exception as exc:
            print(f"Warning: HuggingFace login failed: {exc}")

    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        print(f"Loading model from HuggingFace: {self.model_id}")
        
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for model '{self.model_id}'.\n"
                f"Error: {e}\n"
                f"Please check that the model ID is correct and you have internet access."
            ) from e

        try:
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype="auto",
                device_map="auto",
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                raise RuntimeError(
                    f"Failed to load model '{self.model_id}' - out of memory.\n"
                    f"Error: {e}\n"
                    f"Try using a smaller model or ensure you have enough GPU/RAM."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to load model '{self.model_id}'.\n"
                    f"Error: {e}\n"
                    f"Please check that the model ID is correct and you have sufficient resources."
                ) from e

        print("Model loaded successfully")

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    def get_llm_response(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0,
        response_format: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Get a response from the local LLM model.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            model: Model name (ignored for local models, kept for API compatibility)
            temperature: Sampling temperature (0 = deterministic)
            response_format: Optional dict with "type": "json_object" to extract JSON
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            The generated text response, or None if generation failed
        
        Note:
            The model parameter is ignored - the model specified in __init__ is used.
        """
        try:
            # Apply chat template to format messages
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Tokenize and move to model device
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Prepare generation parameters
            generate_params: Dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
            }

            if temperature > 0:
                generate_params["do_sample"] = True
                generate_params["temperature"] = temperature
            else:
                generate_params["do_sample"] = False

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**generate_params)

            # Extract only the new tokens (response)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            if response_format and response_format.get("type") == "json_object":
                response = extract_json_block(response)

            return response
        except Exception as exc:
            error_msg = str(exc)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print(
                    f"Warning: Local LLM generation failed - likely out of memory.\n"
                    f"Error: {error_msg}\n"
                    f"Try using a smaller model or reducing batch size."
                )
            else:
                print(
                    f"Warning: Local LLM generation failed.\n"
                    f"Error: {error_msg}\n"
                    f"This request will be skipped."
                )
            return None

    def _test_connection(self) -> bool:
        """
        Test that the local LLM is working correctly.
        
        Raises:
            Exception: If the test fails (model not responding correctly)
        """
        print("Testing local LLM connection....")
        test_messages = [{"role": "user", "content": "Say 'test' if you can read this."}]
        response = self.get_llm_response(
            messages=test_messages,
            temperature=0,
            max_new_tokens=64,
        )

        if response and len(response.strip()) > 0:
            print("Local LLM connection test successful")
            return True

        raise RuntimeError(
            "Failed to test local LLM connection. The model did not return a valid response.\n"
            "Please check that:\n"
            "  1. The model loaded correctly\n"
            "  2. You have sufficient memory/GPU resources\n"
            "  3. The model ID is correct"
        )


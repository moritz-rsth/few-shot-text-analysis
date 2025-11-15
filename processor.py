from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from config import FewShotConfig
from llm.base import BaseLLMClient


class Processor:
    """
    Simple processor class for few-shot text analysis.
    
    Processes a dataframe by building prompts, calling the LLM,
    and parsing JSON responses to extract target attributes.
    """

    def prompt(
        self,
        df_row: pd.Series,
        df_examples: pd.DataFrame,
        config: FewShotConfig,
    ) -> List[Dict[str, str]]:
        """
        Build a prompt for a given row using the config.
        
        Args:
            df_row: Row from the processing dataframe
            df_examples: Dataframe with examples for few-shot learning
            config: Few-shot configuration
            
        Returns:
            List of message dicts for the LLM
        """
        return config.to_prompt(df_row, df_examples)

    def _parse_json_response(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Parsed JSON dict, or None if parsing fails
        """
        if not response:
            return None

        # Try to extract JSON block if embedded in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON from response: {response[:100]}...")
            return None

    def process(
        self,
        df_processing: pd.DataFrame,
        df_examples: pd.DataFrame,
        client: BaseLLMClient,
        config: FewShotConfig,
        model: Optional[str] = None,
        temperature: float = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Process each row in the processing dataframe.
        
        For each row:
        1. Build a prompt using the config
        2. Get response from LLM
        3. Parse JSON response
        4. Extract target attributes and set them in the dataframe
        
        Args:
            df_processing: Dataframe to process (will be modified in-place)
            df_examples: Dataframe with examples for few-shot learning
            client: LLM client instance
            config: Few-shot configuration
            model: Optional model name for the LLM client
            temperature: Sampling temperature for LLM
            **kwargs: Additional arguments to pass to get_llm_response
            
        Returns:
            The processing dataframe with target attributes added
        """
        # Make a copy to avoid modifying the original
        df_result = df_processing.copy()

        # Initialize target attribute columns if they don't exist
        for attr in config.target_attributes:
            if attr.name not in df_result.columns:
                df_result[attr.name] = None

        # Check if client is a remote client (for parallel processing)
        from llm.remote import RemoteLLMClient
        is_remote_client = isinstance(client, RemoteLLMClient)
        use_parallel = is_remote_client and config.threads > 1

        if use_parallel:
            # Parallel processing for remote API clients
            return self._process_parallel(
                df_result, df_examples, client, config, model, temperature, **kwargs
            )
        else:
            # Sequential processing (for local clients or threads=1)
            return self._process_sequential(
                df_result, df_examples, client, config, model, temperature, **kwargs
            )

    def _process_sequential(
        self,
        df_result: pd.DataFrame,
        df_examples: pd.DataFrame,
        client: BaseLLMClient,
        config: FewShotConfig,
        model: Optional[str],
        temperature: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Process rows sequentially."""
        for idx, row in df_result.iterrows():
            # Build prompt
            messages = self.prompt(row, df_examples, config)

            # Get response from LLM
            response = client.get_llm_response(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs,
            )

            # Parse JSON response
            parsed = self._parse_json_response(response)

            if parsed:
                # Extract target attributes and set them in the dataframe
                for attr in config.target_attributes:
                    if attr.name in parsed:
                        df_result.at[idx, attr.name] = parsed[attr.name]
            else:
                print(f"Warning: Failed to parse response for row {idx}")

        return df_result

    def _process_parallel(
        self,
        df_result: pd.DataFrame,
        df_examples: pd.DataFrame,
        client: BaseLLMClient,
        config: FewShotConfig,
        model: Optional[str],
        temperature: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Process rows in parallel using ThreadPoolExecutor."""
        def process_row(idx_row):
            idx, row = idx_row
            try:
                # Build prompt
                messages = self.prompt(row, df_examples, config)

                # Get response from LLM
                response = client.get_llm_response(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    **kwargs,
                )

                # Parse JSON response
                parsed = self._parse_json_response(response)

                if parsed:
                    result = {}
                    for attr in config.target_attributes:
                        if attr.name in parsed:
                            result[attr.name] = parsed[attr.name]
                    return (idx, result)
                else:
                    print(f"Warning: Failed to parse response for row {idx}")
                    return (idx, None)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                return (idx, None)

        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=config.threads) as executor:
            futures = {
                executor.submit(process_row, (idx, row)): idx
                for idx, row in df_result.iterrows()
            }

            for future in as_completed(futures):
                result = future.result()
                if result and result[1]:
                    idx, parsed = result
                    for attr_name, value in parsed.items():
                        df_result.at[idx, attr_name] = value

        return df_result


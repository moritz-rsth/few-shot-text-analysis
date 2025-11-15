from __future__ import annotations

import json
import os
import re
import time
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
        save_path: Optional[str] = None,
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
            save_path: Optional path to save results periodically (every 1000 rows)
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
                df_result, df_examples, client, config, model, temperature, save_path, **kwargs
            )
        else:
            # Sequential processing (for local clients or threads=1)
            return self._process_sequential(
                df_result, df_examples, client, config, model, temperature, save_path, **kwargs
            )

    def _process_sequential(
        self,
        df_result: pd.DataFrame,
        df_examples: pd.DataFrame,
        client: BaseLLMClient,
        config: FewShotConfig,
        model: Optional[str],
        temperature: float,
        save_path: Optional[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Process rows sequentially with progress logging."""
        total_rows = len(df_result)
        processed = 0
        start_time = time.time()
        last_save_count = 0
        save_interval = 1000

        print(f"Processing {total_rows} rows sequentially...")
        print("-" * 60)

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

            processed += 1

            # Progress logging
            if processed % 10 == 0 or processed == total_rows:
                elapsed_time = time.time() - start_time
                if processed > 0:
                    avg_time_per_row = elapsed_time / processed
                    remaining_rows = total_rows - processed
                    estimated_remaining = avg_time_per_row * remaining_rows
                    
                    progress_pct = (processed / total_rows) * 100
                    print(
                        f"Progress: {processed}/{total_rows} ({progress_pct:.1f}%) | "
                        f"Elapsed: {self._format_time(elapsed_time)} | "
                        f"ETA: {self._format_time(estimated_remaining)}"
                    )

            # Periodic saving
            if save_path and processed - last_save_count >= save_interval:
                self._save_checkpoint(df_result, save_path, processed)
                last_save_count = processed

        # Final save if save_path is provided
        if save_path:
            self._save_checkpoint(df_result, save_path, processed, final=True)

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Completed! Processed {processed} rows in {self._format_time(total_time)}")
        if processed > 0:
            print(f"Average time per row: {total_time/processed:.2f}s")

        return df_result

    def _process_parallel(
        self,
        df_result: pd.DataFrame,
        df_examples: pd.DataFrame,
        client: BaseLLMClient,
        config: FewShotConfig,
        model: Optional[str],
        temperature: float,
        save_path: Optional[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Process rows in parallel using ThreadPoolExecutor with progress logging."""
        total_rows = len(df_result)
        processed = 0
        start_time = time.time()
        last_save_count = 0
        last_log_time = start_time
        save_interval = 1000
        log_interval = 10  # Log every 10 completed tasks

        print(f"Processing {total_rows} rows in parallel ({config.threads} threads)...")
        print("-" * 60)

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

                processed += 1

                # Progress logging (check periodically to avoid too frequent prints)
                current_time = time.time()
                if processed % log_interval == 0 or processed == total_rows or (current_time - last_log_time) >= 2.0:
                    elapsed_time = current_time - start_time
                    if processed > 0:
                        avg_time_per_row = elapsed_time / processed
                        remaining_rows = total_rows - processed
                        estimated_remaining = avg_time_per_row * remaining_rows
                        
                        progress_pct = (processed / total_rows) * 100
                        print(
                            f"Progress: {processed}/{total_rows} ({progress_pct:.1f}%) | "
                            f"Elapsed: {self._format_time(elapsed_time)} | "
                            f"ETA: {self._format_time(estimated_remaining)}"
                        )
                        last_log_time = current_time

                # Periodic saving
                if save_path and processed - last_save_count >= save_interval:
                    self._save_checkpoint(df_result, save_path, processed)
                    last_save_count = processed

        # Final save if save_path is provided
        if save_path:
            self._save_checkpoint(df_result, save_path, processed, final=True)

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Completed! Processed {processed} rows in {self._format_time(total_time)}")
        if processed > 0:
            print(f"Average time per row: {total_time/processed:.2f}s")

        return df_result

    def _format_time(self, seconds: float) -> str:
        """Format seconds into a human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

    def _save_checkpoint(
        self,
        df_result: pd.DataFrame,
        save_path: str,
        processed: int,
        final: bool = False,
    ) -> None:
        """Save checkpoint of current results."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            
            # Save to CSV
            df_result.to_csv(save_path, index=False)
            
            status = "Final" if final else "Checkpoint"
            print(f"\n[{status}] Saved results to {save_path} ({processed} rows processed)")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint to {save_path}: {e}")


from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pandas as pd


@dataclass
class TargetAttribute:
    """ 
    Represents a target attribute to be scored, e.g. 
        TargetAttribute(
                name="CTR", 
                prompt="Rate on a scale of 0 to 100 how likely it is that a person 
                        clicks on this ad. 0 never, 100 always."
                )
    """
    name: str
    prompt: str


# ============================================================================
# Configuration
# ============================================================================

class FewShotConfig:
    """Main configuration class for few-shot prompting."""
    
    def __init__(
        self,
        target_attributes: List[TargetAttribute],
        icl_attributes: List[str],
        few_shot_examples_attributes: List[str],
        system_instruction: Optional[str] = None,
        nr_few_shot_examples: int = 0,
        threads: int = 1,
    ):
        """
        Initialize few-shot configuration.
        
        Args:
            target_attributes: List of TargetAttribute objects to predict
            icl_attributes: Column names used for in-context learning
            few_shot_examples_attributes: Column names to include in few-shot examples
            system_instruction: System prompt for the LLM
            nr_few_shot_examples: Number of examples to include in each prompt
            threads: Number of parallel threads for API processing (default: 1, only for remote clients)
        """
        self.target_attributes = target_attributes
        self.icl_attributes = icl_attributes
        self.system_instruction = system_instruction
        self.nr_few_shot_examples = nr_few_shot_examples 
        self.few_shot_examples_attributes = few_shot_examples_attributes
        self.threads = threads

    
    def _build_json_schema(self) -> str:
        lines = ['{']
        for attr in self.target_attributes:
            lines.append(f'    "{attr.name}" : <int: 1-100>,')
        lines.append('}')
        return "\n".join(lines)
    
    def _build_target_attributes_prompt(self, df_row):
        lines = []
        for attr in self.target_attributes:
            lines.append(f'Target Attribute: {attr.name}')
            lines.append(f'Prompt: {attr.prompt}')
        lines.append("Give the target attributes for the input below:")
        for name in self.icl_attributes:
            lines.append(f'{name}: {df_row[name]}')
            
        return "\n".join(lines)
    

    def _build_random_few_shot_examples(self, df: pd.DataFrame) -> str:
        """
        Builds a few-shot example block using randomly sampled rows.
        Returns a single formatted string.
        """
        if self.nr_few_shot_examples <= 0:
            return ""   # no examples requested
        
        # Sample rows
        sampled = df.sample(self.nr_few_shot_examples, replace=False)

        lines = []
        lines.append("Below are several correctly labeled examples. " 
                     "Use them as guidance for how the target attribute relates" 
                     "to the input features.")

        for _, row in sampled.iterrows():
            for attr in self.few_shot_examples_attributes:
                lines.append(f"{attr}: {row[attr]}")
            lines.append("")   # blank line between examples

        return "\n".join(lines)

    
    def to_prompt(self, df_row, df) -> List[Dict[str, str]]:
        """
            Generate a prompt for the current df_row based on the config scheme
        """

        json_schema = self._build_json_schema()

        # Build system message
        system_message = {
            "role": "system",
            "content":  (   f"{self.system_instruction}\n"
                            "Always answer in the follwing JSON Format:\n"
                            f"{json_schema}\n"
                            "No explanations. No comments. Only the JSON."
                        )
        }
        
        # Build user message
        target_attributes_prompt = self._build_target_attributes_prompt(df_row=df_row)
        few_shot_examples = self._build_random_few_shot_examples(df=df)
        
        user_message = {
            "role": "user",
            "content":  (   f"{target_attributes_prompt}"
                            "\n\n"
                            f"{few_shot_examples}"
                        )
        }
        
        return [system_message, user_message]

    

    



# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    CTR_llm = TargetAttribute(
        name="CTR_llm",
        prompt=("CTR (Click-Through Rate) measures how often people click on an ad after seeing it."
                "Estimate the CTR from 0 - 100, "
                "where 0 means that nobody out of 100 people that see the ad click on the ad and "
                "100 means everybody out of 100 people that see the ad clicks on the ad.")
    )

    CVR_llm = TargetAttribute(
        name="CVR_llm",
        prompt=("CVR (Conversion Rate) measures how often people complete the desired action after clicking the ad."
                "Estimate the CVR from 0 - 100, "
                "where 0 means that nobody out of 100 people that click on the ad perform the disired action adn "
                "100 means everybody out of 100 people that click the ad performs the desired option.")
    )

    # Case 1: Only CTR Prediction
    CTR_prediction_config = FewShotConfig(
        system_instruction="You're a helpful assistant!",
        target_attributes=[CTR_llm],
        icl_attributes=["ad_title", "image_description", "iab_category"],
        few_shot_examples_attributes=["ad_title", "image_description", "iab_category", "CTR"],
        nr_few_shot_examples=5,
    )


    # Case 2: Predicting both at the same time
    CTR_CVR_prediction_config = FewShotConfig(
        system_instruction="You're a helpful assistant!",
        target_attributes=[CTR_llm, CVR_llm],
        icl_attributes=["ad_title", "image_description", "iab_category"],
        few_shot_examples_attributes=["ad_title", "image_description", "iab_category", "CTR", "CVR"],
        nr_few_shot_examples=5,
    )


    df = pd.read_csv("./data/raw/AdExamples.csv")
    df_examples = df.sample(frac=0.7, random_state=42)
    df_predict = df.drop(df_examples.index)

    rows = df_predict.sample(2, random_state=42)

    prompts = [
        CTR_CVR_prediction_config.to_prompt(row, df_examples)
        for _, row in rows.iterrows()
    ]


    # Write example prompts to file for documentation
    output_file = "example_prompt.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Example Few-Shot Prompts Generated by FewShotConfig\n")
        f.write("=" * 60 + "\n\n")
        f.write("This file demonstrates how prompts are structured when using FewShotConfig.\n")
        f.write("Each prompt consists of a system message and a user message.\n")
        f.write("The system message defines the JSON format, and the user message contains\n")
        f.write("the target attributes, input features, and few-shot examples.\n\n")
        f.write("=" * 60 + "\n\n")
        
        for i, prompt in enumerate(prompts, start=1):
            f.write(f"=== Prompt {i} ===\n\n")

            for msg in prompt:
                role = msg["role"].upper()
                content = msg["content"]

                f.write(f"[{role}]\n")
                f.write("-" * 60 + "\n")
                f.write(content)
                f.write("\n\n")

            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"Example prompts written to {output_file}")



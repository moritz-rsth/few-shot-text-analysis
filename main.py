import pandas as pd
from config import FewShotConfig, TargetAttribute
from processor import Processor
from llm import RemoteLLMClient
from visualize import AccuracyAnalyzer


# Define what to predict
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
    nr_few_shot_examples=1,
    threads=1,
)


# Case 2: Only CVR Prediction
CVR_prediction_config = FewShotConfig(
    system_instruction="You're a helpful assistant!",
    target_attributes=[CVR_llm],
    icl_attributes=["ad_title", "image_description", "iab_category"],
    few_shot_examples_attributes=["ad_title", "image_description", "iab_category", "CVR"],
    nr_few_shot_examples=1,
    threads=1,
)


# Case 3: Predicting both at the same time
CTR_CVR_prediction_config = FewShotConfig(
    system_instruction="You're a helpful assistant!",
    target_attributes=[CTR_llm, CVR_llm],
    icl_attributes=["ad_title", "image_description", "iab_category"],
    few_shot_examples_attributes=["ad_title", "image_description", "iab_category", "CTR", "CVR"],
    nr_few_shot_examples=1,
    threads=1,
)


# Split data into examples and prediction set
df = pd.read_csv("./data/raw/AdExamples.csv")
df_examples = df.sample(frac=0.5, random_state=42)
df_predict = df.drop(df_examples.index)

# Intialize objects
processor = Processor()
client = RemoteLLMClient()
analyser = AccuracyAnalyzer()

# Process predictions
df_analysed = processor.process(
    df_processing=df_predict,
    df_examples=df_examples,
    client=client,
    config=CTR_CVR_prediction_config,
    temperature=0,
    model="meta-llama/Llama-3.1-8B-Instruct",
)

# Calculate accuracy
results = analyser.calculate_accuracy(
    df_analysed=df_analysed,
    df_examples=df_predict,
    target_attributes=["CTR_llm", "CVR_llm"],
    measured_columns=["CTR", "CVR"]
)

analyser.print_results(results)
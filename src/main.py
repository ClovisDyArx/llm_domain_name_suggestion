import json
import os
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI

from inference import generate_suggestions
from evaluation import evaluate_suggestions


def judge_pipeline() -> pd.DataFrame:
    """
    Evaluates the fine-tuned model using a powerful LLM as a judge (gpt4o) on the test dataset.
    """
    # test set
    test_set = []
    with open("data/test_dataset.jsonl", 'r') as f:
        for line in f:
            test_set.append(json.loads(line))

    # judge model
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )

    all_evaluations = []

    # evaluation loop
    for item in tqdm(test_set, desc="Evaluating fine-tuned Model"):
        business_desc = item['business_description']
        
        generated_domains = generate_suggestions(business_desc)
        
        evaluation_score = evaluate_suggestions(client, business_desc, generated_domains)
        
        if evaluation_score:
            all_evaluations.append({
                "business_description": business_desc,
                "generated_domains": generated_domains,
                **evaluation_score
            })

    eval_df = pd.DataFrame(all_evaluations)
    baseline_performance = {
        "avg_relevance": eval_df['relevance_score'].mean(),
        "avg_creativity": eval_df['creativity_score'].mean(),
        "avg_diversity": eval_df['diversity_score'].mean(),
        "avg_overall": eval_df['overall_score'].mean()
    }

    print("\n--- Baseline Model Performance ---")
    print(json.dumps(baseline_performance, indent=2))

    eval_df.to_csv("outputs/evaluation_logs/baseline_v1_evaluation.csv", index=False)
    
    return eval_df


def edge_cases():# -> pd.DataFrame:
    return True


if __name__ == "__main__":
    eval_df = judge_pipeline()
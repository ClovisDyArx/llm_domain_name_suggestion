import json
import os
from tqdm import tqdm
import pandas as pd

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI

from inference import generate_suggestions


def get_judge_prompt_template():
    return """You are an impartial and expert evaluator. Your task is to evaluate the quality of a list of AI-generated domain name suggestions based on a business description.

    You must score the provided list of domain names on three criteria: Relevance, Creativity, and Diversity, each on a scale of 1 to 5, where 1 is poor and 5 is excellent.

    1.  **Relevance**: How closely do the domain names relate to the core business?
    2.  **Creativity**: How inventive and brandable are the names? Are they more than just literal keywords?
    3.  **Diversity**: Does the list include a good mix of TLDs (e.g., .com, .io, .ai) and naming styles?

    Finally, provide an `overall_score` from 1 to 5.

    You MUST return your evaluation as a single, valid JSON object, and nothing else. The JSON object must contain the scores and a brief `justification`.

    Example Input:
    - Business Description: "A subscription box for Japanese snacks and candies."
    - Generated Domains: ["japansnacks.com", "tokyotreats.com", "snackjapan.com", "candynihon.com"]

    Example JSON Output:
    {
    "relevance_score": 5,
    "creativity_score": 2,
    "diversity_score": 1,
    "overall_score": 2,
    "justification": "The domains are highly relevant but lack creativity and diversity. All are .com and use very literal keywords."
    }
    """


def evaluate_suggestions(client, business_description, generated_domains):
    """
    Uses a judge LLM to evaluate the quality of domain suggestions.
    """
    domains_str = ", ".join(generated_domains)
    
    user_prompt = f"""
    - Business Description: "{business_description}"
    - Generated Domains: [{domains_str}]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_judge_prompt_template()},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    

def run_evaluation_pipeline(
    client,
    model,
    tokenizer,
    test_set_path : str,
    output_file : str,
    ):
    test_set = []
    with open(test_set_path, 'r') as f:
        for line in f:
            test_set.append(json.loads(line))
    
    all_evaluations = []

    for item in tqdm(test_set, desc="Evaluating fine-tuned Model"):
        business_desc = item['business_description']
        
        generated_domains = generate_suggestions(model, tokenizer, business_desc)
        
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

    eval_df.to_csv(f"outputs/evaluation_logs/{output_file}", index=False)
    
    return baseline_performance, eval_df


if __name__ == '__main__':
    # exemple d'utilisation
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )
    
    test_description = "A blog about vegan recipes."
    test_domains = ["veganplates.com", "plantifulbites.co", "greenchef.ai"]
    
    evaluation_result = evaluate_suggestions(client, test_description, test_domains)
    print(json.dumps(evaluation_result, indent=2))
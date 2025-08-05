import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from openai import AzureOpenAI

from dataset import create_dataset
from model import DomainLLM
from training import finetune_model
from evaluation import evaluate_suggestions, run_evaluation_pipeline
from inference import generate_suggestions, load_model_and_tokenizer
from safety import is_request_inappropriate
from utils import combine_datasets


def full_pipeline():
    """
    Full end-to-end pipeline for the LLM domain generation project.
    """
    # -------- ENV / VARS SETUP --------
    print("--- Starting Full AI Engineering Pipeline ---")

    ORIGINAL_DATASET_PATH = "data/training_dataset.jsonl"
    TEST_SET_PATH = "data/test_dataset.jsonl"
    EDGE_CASE_DATASET_PATH = "data/edge_case_dataset.jsonl"
    COMBINED_DATASET_PATH = "data/combined_training_dataset.jsonl"
    
    BASE_MODEL_ID = "google/gemma-2-2b-it"
    
    BASELINE_MODEL_DIR = "models/gemma2-baseline-v1"
    IMPROVED_MODEL_DIR = "models/gemma2-improved-v2"
    
    BASELINE_CSV = "gemma2-baseline-v1.csv"
    IMPROVED_CSV = "gemma2-improved-v2.csv"
    
    CLIENT = AzureOpenAI(
        azure_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION")
    )
    
    MODERATION_DEPLOYMENT = os.environ.get("OPENAI_MODERATION_DEPLOYMENT")
    
    # -------- DATASET --------
    # training set
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        print("\n[Step 1] Creating initial training dataset...")
        create_dataset(
            client=CLIENT,
            output_path=ORIGINAL_DATASET_PATH,
            test=False,
            debug=True,
        )
    else:
        print(f"\n[Step 1] Training dataset found at '{ORIGINAL_DATASET_PATH}'. Skipping creation.")
    
    # testing set
    if not os.path.exists(TEST_SET_PATH):
        print("\n[Step 1] Creating initial testing dataset...")
        create_dataset(
            client=CLIENT,
            output_path=TEST_SET_PATH,
            test=True,
            debug=True,
        )
    else:
        print(f"\n[Step 1] Testing dataset found at '{TEST_SET_PATH}'. Skipping creation.")
    
    # -------- BASELINE MODEL FINE-TUNING --------
    if not os.path.exists(os.path.join(BASELINE_MODEL_DIR, "final")):
        print(f"\n[Step 2] Fine-tuning baseline model '{BASE_MODEL_ID}'...")
        finetune_model(
            dataset_path=ORIGINAL_DATASET_PATH,
            base_model_id=BASE_MODEL_ID,
            output_dir=BASELINE_MODEL_DIR
        )
    else:
        print(f"\n[Step 2] Baseline model found at '{BASELINE_MODEL_DIR}'. Skipping training.")
        
    # -------- MODEL LOADING --------
    model, tokenizer = load_model_and_tokenizer(base_model_id=BASE_MODEL_ID, adapter_path=BASELINE_MODEL_DIR+"/final")
        
    # -------- BASELINE MODEL EVALUATION --------
    print("\n[Step 3] Evaluating baseline model performance...")
    baseline_results, df_results = run_evaluation_pipeline(
        client=CLIENT,
        model=model,
        tokenizer=tokenizer,
        test_set_path=TEST_SET_PATH,
        output_file=BASELINE_CSV
    )
    baseline_score = baseline_results['avg_overall']
    print(f"-> Baseline Model Average Overall Score: {baseline_score:.2f}")
    
    # -------- ITERATIVE IMPROVEMENT --------
    print("\n[Step 4] Starting iterative improvement cycle...")
    if not os.path.exists(EDGE_CASE_DATASET_PATH):
        print("-> Edge case dataset not found. Skipping improvement cycle.")
        best_model_dir = BASELINE_MODEL_DIR
    else:
        print("-> Found edge case dataset. Proceeding with improvement.")
        combine_datasets(ORIGINAL_DATASET_PATH, EDGE_CASE_DATASET_PATH, COMBINED_DATASET_PATH)

        if not os.path.exists(os.path.join(IMPROVED_MODEL_DIR, "final")):
            print(f"-> Fine-tuning improved model on combined dataset...")
            finetune_model(
                dataset_path=COMBINED_DATASET_PATH,
                base_model_id=BASE_MODEL_ID,
                output_dir=IMPROVED_MODEL_DIR
            )
            model, tokenizer = load_model_and_tokenizer(base_model_id=BASE_MODEL_ID, adapter_path=IMPROVED_MODEL_DIR+"/final")
        else:
            print(f"-> Improved model found at '{IMPROVED_MODEL_DIR}'. Skipping training.")
        
        print("-> Evaluating improved model performance...")
        improved_results, df_results = run_evaluation_pipeline(
            client=CLIENT,
            model=model,
            tokenizer=tokenizer,
            test_set_path=TEST_SET_PATH,
            output_file=IMPROVED_CSV
        )
        improved_score = improved_results['avg_overall']
        print(f"-> Improved Model Average Overall Score: {improved_score:.2f}")

        # -------- COMPARE MODELS & SELECT BEST --------
        print("\n[Step 5] Comparing model performance...")
        print(f"  - Baseline Score: {baseline_score:.2f}")
        print(f"  - Improved Score: {improved_score:.2f}")
        print(f"  - Improvement: {improved_score - baseline_score:+.2f}")
        best_model_dir = IMPROVED_MODEL_DIR if improved_score > baseline_score else BASELINE_MODEL_DIR

    print(f"-> Best performing model selected: '{best_model_dir}'")

    # -------- SAFETY GUARDRAIL DEMONSTRATION --------
    print("\n[Step 6] Demonstrating safety guardrails...")
    test_cases = {
        "safe_text": "A bakery for dogs that makes custom birthday cakes.",
        "unsafe_text": "A website selling illegal firearms and weapons.",
        "safe_tricky_text": "A horrible bakery for big dogs that makes devishly amazing custom birthday cakes.",
        "unsafe_tricky_text": "A website selling super cool illegal firearms and cute killing weapons."
    }

    for name, text in test_cases.items():
        is_inappropriate = is_request_inappropriate(client, text, MODERATION_DEPLOYMENT)
        print(f"'{text}' is inappropriate: {is_inappropriate}")
    
    # -------- INFERENCE EXAMPLE --------
    print("\n[Step 7] Final inference example with the best model...")
    final_prompt = "A mobile app that uses AI to generate personnalized domain names for any business ideas." # very original I know.
    print(f"\t- Generating domains for: '{final_prompt}'")

    if is_request_inappropriate(CLIENT, final_prompt, MODERATION_DEPLOYMENT):
         print("\t-> Request blocked by safety guardrail.")
    else:
        final_suggestions = generate_suggestions(
            model=model,
            tokenizer=tokenizer,
            business_description=final_prompt,
        )
        print("\t-> Generated Domains:")
        for domain in final_suggestions:
            print(f"\t- {domain}")

    print("\n--- Full Pipeline Complete ---")
    
    
if __name__ == "__main__":
    full_pipeline()
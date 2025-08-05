# LLM for domain name suggestion

AI Engineering - Take home assignment

# Setup

1. **Clone the repository**
    ```bash
    git clone git@github.com:ClovisDyArx/llm_domain_name_suggestion.git
    cd llm-domain-name-suggestion
    ```

2. **Create and activate a Python virtual environment**
    ```bash
    # Ensure you are using a compatible Python version
    python --version
    # > Python 3.10+ recommended

    python -m venv .llm_domain_venv
    source .llm_domain_venv/bin/activate
    ```


3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
    Create a file named `.env` in the root directory and add your API credentials. This is required for the evaluation (LLM-as-a-Judge) and safety guardrail steps.
    ```
    OPENAI_API_KEY="sk-..."
    OPENAI_ENDPOINT="https://..."
    OPENAI_API_VERSION="2024-05-01-preview"
    OPENAI_MODERATION_DEPLOYMENT="your-moderation-deployment-name"
    ```

# Architecture

The project is structured into several core modules:

- `main.py`: The main orchestrator script to run the full end-to-end pipeline.
- `src/`: Contains all core logic.
    - `model.py`: Model class
    - `dataset.py`: Logic for creating the synthetic training dataset.
    - `training.py`: Handles the fine-tuning of the language model using PEFT/LoRA.
    - `evaluation.py`: Evaluates the fine-tuned model using a LLM as a judge.
    - `inference.py`: Logic for loading a trained model adapter and generating domain suggestions.
    - `safety.py`: Implements the safety guardrail for content moderation.
    - `utils.py`: Misc functions.
- `data/`: Stores all datasets (`training_dataset.jsonl`, `test_dataset.jsonl`, etc.).
-   `models/`: Stores the trained model adapters (e.g., `gemma2-baseline-v1`).
-   `reports/`: Contains the final technical report.

# How to Use

The entire project workflow can be executed via the main pipeline script.

**Running the Full Pipeline:**

The `main.py` script is designed to run all major steps of the project sequentially:
1. Creates the initial dataset (if not found).
2. Trains the baseline model (if not found).
3. Evaluates the baseline model.
4. Trains and evaluates an improved model using an edge-case dataset (if found).
5. Compares the models.
6. Demonstrates the safety guardrail.

To run the full process, execute:
```bash
python src/main.py
```

*Logs were kept and pushed to the repo. If you don't have any OpenAI api key, you can check the logs to get an idea of the model.*

**Any component can be used single handedly, check out the python files for usage instructions.**

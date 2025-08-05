# Technical Report: Iterative Improvement of a Domain Name Generation LLM

This report details the methodology, analysis, and results of building and iteratively improving a fine-tuned language model for domain name suggestion.

## 1. Methodology & Initial Results

### Dataset Creation Approach 

The initial training dataset was created synthetically to ensure diversity and quality.  The process involved:
* **Seeding Ideas**: A diverse list of over 100 business concepts was manually curated, covering various industries (tech, e-commerce, local services) and complexity levels. 
* **Teacher LLM Generation**: A powerful "teacher" model (GPT-4o via Azure OpenAI) was prompted to act as a branding expert. For each business idea, it generated a list of 5-7 high-quality, creative, and varied domain name suggestions.
* **Structured Output**: The final dataset was formatted as a `.jsonl` file, with each line containing a JSON object with `business_description` and `domain_suggestions` keys.

### Baseline Model Selection 
* **Model**: The `google/gemma-2-2b-it` model was selected. This choice was based on its strong performance as an instruction-tuned model, its manageable size for local fine-tuning, and its open-source nature, which was a requirement. Multilingual capabilities are a plus that was taken into consideration aswell.
* **Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT) using the LoRA (Low-Rank Adaptation) method was employed.  This allowed for efficient training by only updating a small subset of the model's parameters.

### Initial Model Performance 
The baseline model was evaluated using an LLM-as-a-Judge framework on a held-out test set.  The judge (GPT-4o) scored outputs on a 1-5 scale across four criteria. The initial results were:

| Metric          | Average Score |
| --------------- | ------------- |
| Relevance       | 4.77          |
| Creativity      | 3.72          |
| Diversity       | 3.63          |
| **Overall** | **4.01** |

## 2. Edge Case Analysis

### Discovery Process 
Edge cases were systematically discovered through manual, adversarial probing of the baseline model.  Prompts were designed to test specific weaknesses, including:
* Highly technical or niche jargon.
* Ambiguous or vague business descriptions.
* Descriptions containing negative constraints (e.g., "a pet store that does *not* sell live animals").
* Abstract or philosophical concepts.
* Languages other than english (French)

### Failure Taxonomy 
Failures observed during testing were categorized into a taxonomy: 
* **Ignoring Constraints**: The model failed to adhere to negative constraints. For example, when prompted for a pet supply store with no live animals, it suggested domains like `happypuppyhome.com`.
* **Literalness**: For highly technical prompts, the model often resorted to simply concatenating keywords, resulting in uncreative and clunky domain names.
* **Formatting Errors**: The model failed to produce a clean, comma-separated list, instead generating a changing markdown format. A work on answer parsing was necessary to reduce to reduce the errors.

### Frequency Analysis 
During manual testing of approximately 50 adversarial prompts, "Formatting Errors" was the most common failure mode, followed by "Ignoring Constraints" on technical topics.

## 3. Iterative Improvement

### Improvement Strategies 
The primary strategy for improvement was **dataset augmentation**.  Based on the edge case analysis, a new dataset (`edge_case_dataset.jsonl`) was created. This dataset contained new training examples that explicitly demonstrated the correct behavior for the identified failure modes. The original and edge case datasets were then combined to create a new, more robust training set for the next iteration. If I had more time, I would also try other SLM/LLM and see which one are more reliable.

### LLM Judge Validation 
To ensure the quality and consistency of the evaluation, the following steps were taken:
* **Powerful Judge Model**: A state-of-the-art model (GPT-4o) was used as the judge to ensure high-quality, nuanced evaluations. 
* **Detailed Rubric**: A clear and systematic scoring rubric with definitions for each criterion was provided to the judge in every call. 
* **Consistent Prompting**: The evaluation prompt was kept identical across all model versions to ensure a fair comparison.

## 4. Model Comparison & Recommendations

### Production Readiness 
I would recommend deploying the **baseline model**. Its performance on edge cases and overall robustness and reliability throughout the project is suited for a user-facing application. Furthermore, its integration with the safety guardrail, which blocks inappropriate requests, is a critical component for production deployment. 

### Future Improvements 
To continue improving the model, the following steps could be taken:
* **Continuous Data Augmentation**: Continuously add new, challenging examples to the training data as new failure modes are discovered in production.
* **Check Domain Availability**: Integrate a post-processing step that uses an API to check the real-world availability of the suggested domains, providing more immediate value to the user.
* **Hyperparameter Optimization**: Conduct a more exhaustive search for optimal fine-tuning hyperparameters (e.g., learning rate, LoRA rank) to potentially boost performance further.
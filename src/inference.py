import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from utils import format_prompt, format_answer_gemma2

# we create placeholders for better performances.
_model = None
_tokenizer = None


def load_model_and_tokenizer(base_model_id : str, adapter_path : str):
    """
    Loads the base model and LoRA adapter. Caches them globally.
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    # quantization so that it runs on lower end gpus.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # loading the base model (not fine-tuned)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # loading the tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # loading the peft model into the fine-tuned model
    _model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Model and tokenizer loaded successfully.")
    return _model, _tokenizer


def generate_suggestions(
    model,
    tokenizer,
    business_description: str
    ) -> list[str]:
    """
    Generates domain name suggestions for a given business description.
    """
    # same prompt as in training
    prompt = format_prompt(business_description)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    
    result_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_only_generated = result_full.split("<start_of_turn>model\n")[-1]


    suggestions = format_answer_gemma2(result_only_generated)  
    return suggestions


if __name__ == '__main__':
    # exemple d'utilisation
    base_model_id = "google/gemma-2-2b-it"
    adapter_path = "../models/gemma2-baseline-v1/final"
    model, tokenizer = load_model_and_tokenizer(base_model_id, adapter_path)
    
    test_description = "A sustainable fashion brand that uses recycled ocean plastic."
    suggestions = generate_suggestions(model, tokenizer, test_description)
    
    print(f"Business idea: '{test_description}'")
    print("\nGenerated Suggestions:")
    for s in suggestions:
        print(f"- {s}")
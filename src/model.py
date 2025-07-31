import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class DomainLLM:
    def __init__(self, model_id : str = "google/gemma-2-2b-it"):
        """
        Class containing all the information relevant to the model creation.
        """
        # default at gemma-2.2b instruct
        self.model_id = model_id
        
        # quantization so that it runs on lower end gpus.
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # we load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side
        self.tokenizer = tokenizer
        
        # we load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        self.full_model = model
        
        # param efficient fine-tuning: we only train the new low rank matrices.
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.peft_model = get_peft_model(model, lora_config)
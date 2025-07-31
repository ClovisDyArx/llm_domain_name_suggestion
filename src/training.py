from transformers import TrainingArguments
from trl import SFTTrainer
import json

from model import DomainLLM
from dataset import DomainDataset


def train_baseline_model():
    """
    Main function to fine-tune the baseline model.
    """
    
    # Instanciate the model's class
    domain_llm = DomainLLM(model_id="google/gemma-2-2b-it")
    
    # Instanciate the dataset's class
    domain_dataset = DomainDataset(data_files="data/training_dataset.jsonl")

    # Config training arguments
    output_dir = "models/gemma2-baseline-v1"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    # Config SFT trainer
    trainer = SFTTrainer(
        model=domain_llm.peft_model,
        train_dataset=domain_dataset.formatted_dataset,
        #dataset_text_field="text",
        #max_seq_length=512,
        args=training_args,
        #tokenizer=domain_llm.tokenizer,
    )

    print("Starting baseline model training...")
    trainer.train()
    print("Training complete.")

    # Saving model
    final_model_path = f"{output_dir}/final"
    trainer.save_model(final_model_path)
    print(f"Baseline model saved to {final_model_path}")


if __name__ == '__main__':
    train_baseline_model()
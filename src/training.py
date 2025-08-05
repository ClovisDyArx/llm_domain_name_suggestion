from transformers import TrainingArguments
from trl import SFTTrainer
import json

from model import DomainLLM
from dataset import DomainDataset


def finetune_model(
    dataset_path : str,
    base_model_id : str,
    output_dir : str,
    epochs : int  = 100,
    log_steps : int = 20,
):
    """
    Main function to fine-tune the baseline model.
    """
    
    # instanciate the model's class
    domain_llm = DomainLLM(model_id=base_model_id)
    
    # instanciate the dataset's class
    domain_dataset = DomainDataset(data_files=dataset_path)

    # config training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        logging_steps=log_steps,
        save_strategy="epoch",
        fp16=False,
    )

    # config SFT trainer
    trainer = SFTTrainer(
        model=domain_llm.peft_model,
        train_dataset=domain_dataset.formatted_dataset,
        args=training_args,
    )

    print("Starting baseline model training...")
    trainer.train()
    print("Training complete.")

    # saving model
    final_model_path = f"{output_dir}/final"
    trainer.save_model(final_model_path)
    print(f"Baseline model saved to {final_model_path}")


if __name__ == '__main__':
    # exemple d'utilisation
    ORIGINAL_DATASET_PATH = "data/training_dataset.jsonl"
    BASE_MODEL_ID = "google/gemma-2-2b-it"
    BASELINE_MODEL_DIR = "models/gemma2-baseline-v1"
    
    finetune_model(
        dataset_path=ORIGINAL_DATASET_PATH,
        base_model_id=BASE_MODEL_ID,
        output_dir=BASELINE_MODEL_DIR
    )
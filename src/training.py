import os
import argparse
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk

def train_model(language_pair, data_dir, output_dir, epochs=3, batch_size=2):
    # Load processed dataset
    tokenized_dataset = load_from_disk(data_dir)
    
    # Parse language pair
    src_lang, tgt_lang = language_pair.split('-')
    
    # Load model and tokenizer
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Define training arguments with memory optimizations
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Accumulate gradients
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        fp16=True,  # Use mixed precision
        dataloader_num_workers=0,  # Reduce parallelism
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    model_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    return model_output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a translation model")
    parser.add_argument("--language-pair", type=str, default="de-en", help="Language pair (e.g., de-en)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with processed data")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    
    args = parser.parse_args()
    
    train_model(args.language_pair, args.data_dir, args.output_dir, args.epochs, args.batch_size)

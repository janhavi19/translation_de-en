import os
from datasets import load_dataset
from transformers import MarianTokenizer

def download_dataset(language_pair, output_dir):
    """Download a parallel corpus dataset."""
    dataset = load_dataset("opus100", language_pair)
    # Save metadata about the dataset
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "info.txt"), "w") as f:
        f.write(f"Dataset: opus100\nLanguage pair: {language_pair}\n")
    return dataset

def tokenize_dataset(dataset, model_name, max_length=128):
    """Tokenize a parallel corpus dataset."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # First, let's check what keys are available in the dataset
    print("Available keys in the dataset:", list(dataset['train'].features.keys()))
    
    # Get the first example to see the structure
    print("First example:", dataset['train'][0])
    
    # Based on opus100 structure, the keys are likely just "translation"
    # with nested source and target keys
    
    def tokenize_pairs(examples):
        # Extract source and target texts correctly
        if "translation" in examples:
            # Handle the case where data is in "translation" dict with language keys
            translations = examples["translation"]
            source_texts = [item["en"] for item in translations]
            target_texts = [item["de"] for item in translations]
        elif "en" in examples and "de" in examples:
            # Handle the case with direct language keys
            source_texts = examples["en"]
            target_texts = examples["de"]
        else:
            # If neither structure is found, print keys and raise error
            print("Available keys:", examples.keys())
            raise ValueError("Could not find expected keys in the dataset")
        
        source_tokenized = tokenizer(source_texts, padding="max_length", 
                                    truncation=True, max_length=max_length)
        target_tokenized = tokenizer(target_texts, padding="max_length", 
                                   truncation=True, max_length=max_length)
        
        return {
            "input_ids": source_tokenized["input_ids"],
            "attention_mask": source_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"]
        }
    
    tokenized_dataset = dataset.map(tokenize_pairs, batched=True)
    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Process translation data")
    parser.add_argument("--language-pair", type=str, default="de-en", 
                       help="Language pair (e.g., de-en)")
    args = parser.parse_args()
    
    language_pair = args.language_pair
    
    # Use the correct model name format
    # For "de-en" language pair, use "Helsinki-NLP/opus-mt-de-en"
    model_name = f"Helsinki-NLP/opus-mt-{language_pair}"
    output_dir = os.path.join("data", "processed", language_pair)
    
    # Download and process the dataset
    dataset = download_dataset(language_pair, output_dir)
    tokenized_dataset, tokenizer = tokenize_dataset(dataset, model_name)
    
    # Save the processed dataset
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
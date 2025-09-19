from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from PIL import Image

def preprocess_data(example):
    """Preprocess images and text for BLIP."""
    image = Image.open(example["image"]).convert("RGB")
    text = example["text"]

    # Tokenize input for BLIP
    inputs = processor(images=image, text=text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    return {"input_ids": inputs["input_ids"].squeeze(0), "pixel_values": inputs["pixel_values"].squeeze(0)}

if __name__ == "__main__":
    print("✅ Loading Dataset...")
    dataset = load_dataset("json", data_files={"train": "annotations_fixed.json"})

    print("✅ Loading BLIP Model & Processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", force_download=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", force_download=True)

    print("✅ Preprocessing Dataset...")
    dataset = dataset.map(preprocess_data, remove_columns=["image", "text"])

    print("✅ Defining Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./blip_sensitive_leakage",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,  # Adjust based on GPU
        per_device_eval_batch_size=8,
        num_train_epochs=3,  # Start with 3 epochs
        save_total_limit=2,
        fp16=True,  # Enable mixed precision for faster training
        logging_dir="./logs"
    )

    print("✅ Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=processor.tokenizer
    )

    print("🚀 Starting Fine-Tuning...")
    trainer.train()

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from PIL import Image
from peft import LoraConfig, get_peft_model
import os

# ✅ Set Cache Directory
CACHE_DIR = "./cached_dataset"


# ✅ Step 1: Define Dataset Preprocessing
def preprocess_data(batch):
    """Processes images and text for BLIP in smaller batches."""
    processed_batch = {"input_ids": [], "pixel_values": [], "labels": []}

    for image_path, text in zip(batch["image"], batch["text"]):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt", padding="max_length", max_length=128,
                           truncation=True)

        processed_batch["input_ids"].append(inputs["input_ids"].squeeze(0))
        processed_batch["pixel_values"].append(inputs["pixel_values"].squeeze(0))
        processed_batch["labels"].append(inputs["input_ids"].squeeze(0))  # 👈 This is the key line

    return processed_batch


if __name__ == "__main__":
    print("✅ Freeing up disk space...")

    # ✅ Step 2: Delete Hugging Face Cache (To free disk space)
    os.system("rmdir /S /Q C:\\Users\\yuval\\.cache\\huggingface")

    # ✅ Step 3: Load Dataset & Limit to First 20k per Category
    print("✅ Loading Dataset...")

    if os.path.exists(CACHE_DIR):
        print("✅ Loading Cached Dataset...")
        dataset = DatasetDict.load_from_disk(CACHE_DIR)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    else:
        print("✅ Loading Annotations JSON...")
        full_dataset = load_dataset("json", data_files={"train": "annotations_fixed.json"})["train"]

        # Extract category from path
        def get_category(example):
            return example["image"].split("/")[2]

        full_dataset = full_dataset.map(lambda x: {"category": get_category(x)})

        categories = ["credit_card", "medical_letter", "phone_bills", "pin_code"]
        limited_datasets = []

        for cat in categories:
            print(f"✅ Selecting 20k from category: {cat}")
            cat_subset = full_dataset.filter(lambda x: x["category"] == cat).select(range(20000))
            limited_datasets.append(cat_subset)

        # Combine and shuffle
        balanced_dataset = concatenate_datasets(limited_datasets).shuffle(seed=42)
        dataset = balanced_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print(f"✅ Balanced Dataset Loaded. Training: {len(train_dataset)}, Validation: {len(eval_dataset)}")

        # ✅ Step 4: Load BLIP Processor
        print("✅ Loading BLIP Processor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        # ✅ Step 5: Preprocess & Save Dataset (Runs only once)
        print("✅ Preprocessing Dataset (This will take time once)...")
        train_dataset = train_dataset.map(preprocess_data, remove_columns=["image", "text", "category"], batched=True,
                                          batch_size=500)
        eval_dataset = eval_dataset.map(preprocess_data, remove_columns=["image", "text", "category"], batched=True,
                                        batch_size=500)

        # ✅ Save preprocessed dataset to disk
        dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})
        dataset.save_to_disk(CACHE_DIR)
        print("✅ Preprocessed Dataset Saved for Future Runs.")

    # ✅ Step 6: Load BLIP Model & Apply LoRA for Low VRAM (6GB GPUs)
    print("✅ Loading BLIP Model & Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["mlp.fc1", "mlp.fc2"],  # Tune only small parts of the model
        lora_dropout=0.1,
        bias="none"
    )

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ✅ Step 7: Define Training Arguments
    print("✅ Defining Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./blip_sensitive_leakage",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_total_limit=2,
        fp16=True,
        logging_dir="./logs"
    )

    # ✅ Step 8: Initialize Trainer
    print("✅ Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer
    )

    # ✅ Step 9: Start Fine-Tuning
    print("🚀 Starting Fine-Tuning...")
    trainer.train()

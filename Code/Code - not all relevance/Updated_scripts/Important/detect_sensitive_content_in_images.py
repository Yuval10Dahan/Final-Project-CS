from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


def run_inference():
    print("Load fine-tuned model and base processor")

    # Load processor from base model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load fine-tuned model from checkpoint
    checkpoint_path = "./blip_sensitive_leakage/checkpoint-54000"
    model = BlipForConditionalGeneration.from_pretrained(checkpoint_path)

    # Load an image
    print("Loading test image")

    # Bad example
    # testing_image_path = "./images/Generic_letter/letter_1.png"

    # Good example
    testing_image_path = "./images/test/test.jpg"
    image = Image.open(testing_image_path).convert("RGB")

    # Move to device
    print("Moving model to device")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=image, return_tensors="pt").to(device)
    model = model.to(device)

    # Generate caption
    print("Generating caption...")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    print("Generated Caption:", caption)


if __name__ == "__main__":
    run_inference()

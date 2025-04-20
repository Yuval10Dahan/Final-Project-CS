from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


def run_inference():
    print("Load processor from base BLIP")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load fine-tuned model
    print("Load fine-tuned model")
    fine_tuned_path = "./blip_sensitive_leakage/checkpoint-54000"
    fine_model = BlipForConditionalGeneration.from_pretrained(fine_tuned_path)

    # Load base model for comparison
    print("Load base BLIP model")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load test image
    print("Load test image")

    # Bad examples
    # image_path = "./images/Generic_letter/letter_1.png"
    # image_path = "./images/Generic_letter/letter_2.png"

    # Good examples
    # image_path = "./images/credit_card/diverse_credit_card_0.png"
    # image_path = "./images/medical_letter/letter_0.png"
    # image_path = "./images/pin_code/email_0.png"
    image_path = "./images/test/test.jpg"
    image = Image.open(image_path).convert("RGB")

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    fine_model.to(device)

    # Preprocess input (no text prompt)
    inputs = processor(images=image, text="", return_tensors="pt").to(device)

    # Caption from base model
    print("Generating caption from base model...")
    base_output = base_model.generate(**inputs, max_new_tokens=50, num_beams=3)
    base_caption = processor.decode(base_output[0], skip_special_tokens=True)

    # Caption from fine-tuned model
    print("Generating caption from fine-tuned model...")
    fine_output = fine_model.generate(**inputs, max_new_tokens=50, num_beams=3)
    fine_caption = processor.decode(fine_output[0], skip_special_tokens=True)

    # Output results
    print("\n Caption Comparison:")
    print(f"Base Model:       {base_caption}")
    print(f"Fine-Tuned Model: {fine_caption}")


if __name__ == "__main__":
    run_inference()

import os
import json
import re
import pytesseract
from PIL import Image

# Set the path to Tesseract (only needed for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\yuval\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Define dataset path
dataset_path = "./images"

# Function to extract numeric key from filenames for proper sorting
def get_numeric_key(filename):
    """ Extract numeric value from filename for correct numerical sorting. """
    numbers = re.findall(r'\d+', filename)  # Extract all numbers from filename
    return int(numbers[0]) if numbers else float('inf')  # Convert first number to int

# Prepare dataset annotation
annotations = []

# Traverse the dataset folders
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue  # Skip if not a folder

    # Sort filenames numerically
    sorted_filenames = sorted(
        [f for f in os.listdir(category_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=get_numeric_key
    )

    for img_name in sorted_filenames:
        img_path = os.path.join(category_path, img_name)

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Perform OCR
            extracted_text = pytesseract.image_to_string(image).strip()

            # If text is empty, use a default category-based description
            if not extracted_text:
                extracted_text = f"This image contains sensitive information related to {category.replace('_', ' ')}."

            # Append annotation
            annotations.append({"image": img_path, "text": extracted_text})

            print(f"✅ Processed: {img_name} (Category: {category})")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

# Sort annotations by image filename numerically
annotations.sort(key=lambda x: get_numeric_key(os.path.basename(x["image"])))

# Save annotations to a JSON file
with open("annotations.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=4)

print(f"\n✅ Dataset annotations.json created with {len(annotations)} image-text pairs.")

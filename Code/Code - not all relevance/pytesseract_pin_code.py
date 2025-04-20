import os
import json
import pytesseract
from PIL import Image
import time


# Path to the folder containing images
image_folder = "pin_code_images"
output_file = "json_files/annotations_pin_code.json"

# Ensure Tesseract is installed and configure its path if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\yuval\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# Function to extract text from an image
def extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()  # Remove unnecessary whitespace
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""


# Function to extract numeric part of the filename for sorting
def get_numeric_key(filename):
    # Extract numeric part from "email_xxx.png"
    return int(filename.split("_")[1].split(".")[0])


start_time = time.perf_counter()  # Start timing

# Process all images in the folder
annotations = []

# Sort the filenames numerically
file_list = sorted(
    [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    key=get_numeric_key
)

for filename in file_list:
    image_path = os.path.join(image_folder, filename)
    print(f"Processing {filename}...")
    text = extract_text(image_path)
    annotations.append({"image_path": image_path, "text": text})

# Save annotations to a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=4, ensure_ascii=False)

print(f"Annotation completed. Saved to {output_file}")

end_time = time.perf_counter()  # End timing
print(f"Total runtime: {end_time - start_time:.2f} seconds")
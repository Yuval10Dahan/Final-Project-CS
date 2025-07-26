import json
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Load fonts
FONT_DIR = "fonts"
FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(".ttf")]

# Output directory for images
OUTPUT_DIR = "images/pin_code"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON dataset
json_file = "json_files/sensitive_bank_emails.json"
with open(json_file, "r") as f:
    data = json.load(f)


# Function to add distortions (optional)
def distort_image(image):
    img = np.array(image)

    # Apply slight Gaussian blur
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 1)

    # Apply random noise
    if random.random() < 0.3:
        noise = np.random.randint(0, 50, img.shape, dtype='uint8')
        img = cv2.add(img, noise)

    # Apply slight rotation
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(img)


# Function to wrap text within the image width
def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_width = draw.textbbox((0, 0), test_line, font=font)[2]  # Get width from bounding box

        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# Function to generate email images
def generate_email_image(email_data, index):
    # Select random font and size
    font_path = random.choice(FONTS)
    font_size = random.randint(22, 26)
    font = ImageFont.truetype(font_path, font_size)

    # Set base dimensions and padding
    max_width = 800
    padding = 40
    line_spacing = font_size + 5  # Dynamic spacing

    # Prepare text content
    email_text = f"{email_data['Email Subject']}\n\n{email_data['Email Body']}"
    text_lines = email_text.split("\n")

    # Create a temporary blank image to calculate height
    temp_image = Image.new("RGB", (max_width, 1000), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_image)

    wrapped_lines = []
    for line in text_lines:
        wrapped_lines.extend(wrap_text(temp_draw, line, font, max_width - padding * 2))

    # Calculate final height
    height = padding * 2 + len(wrapped_lines) * line_spacing
    height = max(height, 250)  # Ensure a minimum height

    # Create final image
    image = Image.new("RGB", (max_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Draw text
    x, y = padding, padding
    for line in wrapped_lines:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_spacing

    # Apply distortions
    image = distort_image(image)

    # Save image
    image_path = os.path.join(OUTPUT_DIR, f"email_{index}.png")
    image.save(image_path)


# Generate images from JSON dataset
for i, email in enumerate(data[:]):  # Generate 1000 images for now
    generate_email_image(email, i)
    print(f"image {i} is generated")

print(f"Generated {min(1000, len(data))} email images in '{OUTPUT_DIR}'")

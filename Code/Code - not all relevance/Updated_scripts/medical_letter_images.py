import json
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load fonts (Ensure you have a "fonts" directory with .ttf font files)
FONT_DIR = "fonts"
FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(".ttf")]

# Output directory for images
OUTPUT_DIR = "images/medical_letter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON dataset
json_file = "json_files/sensitive_medical_letters.json"
with open(json_file, "r") as f:
    data = json.load(f)


# Function to generate a random background color
def get_random_background():
    """Returns a random light-colored background for text images."""
    background_types = ["solid", "gradient"]

    if random.choice(background_types) == "solid":
        return tuple(np.random.randint(200, 255, 3))  # Light pastel solid colors
    else:
        return "gradient"  # Will generate gradient background


# Function to apply a gradient background
def apply_gradient_background(image, start_color, end_color):
    """Applies a vertical gradient background."""
    width, height = image.size
    gradient = Image.new("RGB", (width, height), start_color)

    for y in range(height):
        blend = y / height
        r = int((1 - blend) * start_color[0] + blend * end_color[0])
        g = int((1 - blend) * start_color[1] + blend * end_color[1])
        b = int((1 - blend) * start_color[2] + blend * end_color[2])
        ImageDraw.Draw(gradient).line([(0, y), (width, y)], fill=(r, g, b))

    return gradient


# Function to ensure text contrast
def get_text_color(background_color):
    """Ensures text is readable based on background brightness."""
    if background_color == "gradient":
        return (0, 0, 0)  # Always black on gradient backgrounds
    brightness = sum(background_color) / 3
    return (0, 0, 0) if brightness > 180 else (255, 255, 255)


# Function to wrap text within the image width
def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_width = draw.textbbox((0, 0), test_line, font=font)[2]

        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# Function to generate letter images with different background colors
def generate_letter_image(letter_text, index):
    # Select a random font and size
    font_path = random.choice(FONTS)
    font_size = random.randint(22, 28)
    font = ImageFont.truetype(font_path, font_size)

    # Base dimensions and padding
    max_width = 900
    padding = 50
    line_spacing = font_size + 5

    # Generate a random background color
    background_color = get_random_background()

    # Create a temporary blank image to calculate height
    temp_image = Image.new("RGB", (max_width, 2000), (255, 255, 255))  # Large temp image
    temp_draw = ImageDraw.Draw(temp_image)

    wrapped_lines = wrap_text(temp_draw, letter_text, font, max_width - padding * 2)

    # Calculate the required height based on the number of lines
    height = padding * 2 + len(wrapped_lines) * line_spacing
    height = max(height, 250)

    # Create the final image with the selected background
    if background_color == "gradient":
        start_color = tuple(np.random.randint(200, 255, 3))
        end_color = tuple(np.random.randint(180, 230, 3))
        image = apply_gradient_background(Image.new("RGB", (max_width, height)), start_color, end_color)
    else:
        image = Image.new("RGB", (max_width, height), background_color)

    draw = ImageDraw.Draw(image)

    # Determine text color for readability
    text_color = get_text_color(background_color)

    # Draw text with proper spacing
    x, y = padding, padding
    for line in wrapped_lines:
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_spacing

    # Apply distortions for augmentation
    image = distort_image(image)

    # Save image
    image_path = os.path.join(OUTPUT_DIR, f"letter_{index}.png")
    image.save(image_path)


# Function to apply image distortions
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


# Generate images from JSON dataset
for i, letter_text in enumerate(data[:]):  # Generate 1000 images for now
    generate_letter_image(letter_text, i)
    print(f"image {i} is generated")

print(f"Generated {min(1000, len(data))} letter images in '{OUTPUT_DIR}'")

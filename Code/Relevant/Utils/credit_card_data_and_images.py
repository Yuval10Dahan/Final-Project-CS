import os
import random
import numpy as np
import cv2
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Set up Faker for generating fake names and numbers
fake = Faker()

# Directory to save generated images
output_dir = "./images/credit_card"
os.makedirs(output_dir, exist_ok=True)

# Define colors and variations for card designs
card_colors = [
    "#2E86C1", "#28B463", "#D4AC0D", "#A569BD", "#E74C3C", "#1ABC9C", "#F39C12", "#9B59B6",
    "#34495E", "#E67E22", "#C0392B", "#16A085", "#8E44AD", "#2C3E50", "#BDC3C7", "#95A5A6"
]
font_sizes = [30, 35, 40]
angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
background_colors = [
    "#F0E68C", "#D3D3D3", "#8B4513", "#808080", "#FAEBD7", "#D2B48C", "#A9A9A9", "#696969",
    "#FFFFFF", "#FFFAF0", "#EEE8AA", "#B0C4DE", "#778899", "#708090", "#2F4F4F", "#191970"
]

# Load font (ensure a valid font is available)
# font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_path = "./fonts/Roboto-Italic-VariableFont_wdth,wght.ttf"

# Function to generate a more diverse fake credit card image
def generate_diverse_fake_credit_card(image_id):
    # Select random background color
    bg_color = random.choice(background_colors)
    bg_img = Image.new("RGB", (800, 600), color=bg_color)  # Larger background

    # Create card with random properties
    card = Image.new("RGB", (600, 350), color=random.choice(card_colors))
    draw = ImageDraw.Draw(card)

    # Generate fake card details
    card_number = " ".join([str(random.randint(1000, 9999)) for _ in range(4)])
    name = fake.name()
    expiry = f"{random.randint(1, 12):02d}/{random.randint(24, 30)}"

    # Load font
    font = ImageFont.truetype(font_path, random.choice(font_sizes))

    # Draw text on card
    draw.text((50, 150), card_number, fill="white", font=font)
    draw.text((50, 200), name, fill="white", font=font)
    draw.text((50, 250), f"EXP: {expiry}", fill="white", font=font)

    # Random rotation & resizing
    angle = random.choice(angles)
    card = card.rotate(angle, expand=True)

    # Resize card for distance variation (simulate close-up and far shots)
    scale_factor = random.uniform(0.7, 1.0)  # Adjust scaling to prevent out-of-frame
    new_size = (int(card.width * scale_factor), int(card.height * scale_factor))
    card = card.resize(new_size, Image.Resampling.LANCZOS)

    # Ensure the card fits within the background
    max_x_offset = bg_img.width - new_size[0]
    max_y_offset = bg_img.height - new_size[1]
    x_offset = random.randint(0, max_x_offset)
    y_offset = random.randint(0, max_y_offset)

    # Paste card onto background
    bg_img.paste(card, (x_offset, y_offset), card if card.mode == 'RGBA' else None)

    # Convert to OpenCV format and add blur for realism
    cv_img = np.array(bg_img)
    if random.random() > 0.7:  # Apply blur to some images
        cv_img = cv2.GaussianBlur(cv_img, (5, 5), 0)

    # Convert back to PIL and save
    bg_img = Image.fromarray(cv_img)
    img_path = os.path.join(output_dir, f"diverse_credit_card_{image_id}.png")
    bg_img.save(img_path)

# Generate images
def generate_dataset(num_images):
    for i in range(num_images):
        generate_diverse_fake_credit_card(i)
        print(f"image {i} was generated")

# Generate images for testing
generate_dataset(50000)

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
OUTPUT_DIR = "images/phone_bills"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON dataset
json_file = "json_files/sensitive_phone_bills.json"
with open(json_file, "r") as f:
    data = json.load(f)


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


# Function to generate phone bill images with white background
def generate_bill_image(bill_data, index):
    # Select a random font and size
    font_path = random.choice(FONTS)
    font_size = random.randint(18, 24)
    font = ImageFont.truetype(font_path, font_size)

    # Base dimensions and padding
    max_width = 900
    padding = 50
    line_spacing = font_size + 5

    # Convert JSON bill into readable text
    bill_text = f"""
    {bill_data['Invoice Information']['Carrier']}
    {bill_data['Invoice Information']['Invoice Number']}
    Billing Period: {bill_data['Invoice Information']['Billing Period']}

    Customer Information:
    Name: {bill_data['Customer Information']['Name']}
    ID: {bill_data['Customer Information']['ID']}
    Address: {bill_data['Customer Information']['Address']}
    Phone: {bill_data['Customer Information']['Phone Number']}

    Charges Summary:
    Monthly Service: {bill_data['Charges Summary']['Monthly Service Charge']}
    Call Charges: {bill_data['Charges Summary']['Call Charges']}
    SMS Charges: {bill_data['Charges Summary']['SMS Charges']}
    Data Usage: {bill_data['Charges Summary']['Data Usage']}
    Taxes: {bill_data['Charges Summary']['Taxes & Fees']}
    Late Fee: {bill_data['Charges Summary']['Late Fee']}
    Discount: {bill_data['Charges Summary']['Discount']}
    Total Due: {bill_data['Charges Summary']['Total Due']}

    Payment Information:
    {bill_data['Payment Information']['Due Date']}
    Payment Status: {bill_data['Payment Information']['Payment Status']}
    Bank Name: {bill_data['Payment Information']['Bank Name']}
    Account Number: {bill_data['Payment Information']['Bank Account Number']}
    """

    # Create a temporary blank image to calculate height
    temp_image = Image.new("RGB", (max_width, 2000), (255, 255, 255))  # Large white temp image
    temp_draw = ImageDraw.Draw(temp_image)

    wrapped_lines = wrap_text(temp_draw, bill_text, font, max_width - padding * 2)

    # Calculate the required height based on the number of lines
    height = padding * 2 + len(wrapped_lines) * line_spacing
    height = max(height, 250)

    # Create the final image with a **white background**
    image = Image.new("RGB", (max_width, height), (255, 255, 255))  # PURE WHITE BACKGROUND
    draw = ImageDraw.Draw(image)

    # Text color is **always black**
    text_color = (0, 0, 0)

    # Draw text with proper spacing
    x, y = padding, padding
    for line in wrapped_lines:
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_spacing

    # Apply distortions for augmentation
    image = distort_image(image)

    # Save image
    image_path = os.path.join(OUTPUT_DIR, f"bill_{index}.png")
    image.save(image_path)


# Function to apply image distortions (for realism)
def distort_image(image):
    img = np.array(image)

    # Apply slight Gaussian blur (10% probability)
    if random.random() < 0.1:
        img = cv2.GaussianBlur(img, (5, 5), 1)

    # Apply random noise (15% probability)
    if random.random() < 0.15:
        noise = np.random.randint(0, 50, img.shape, dtype='uint8')
        img = cv2.add(img, noise)

    # Apply slight rotation (20% probability)
    if random.random() < 0.2:
        angle = random.uniform(-3, 3)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(img)


# Generate images from JSON dataset
for i, bill_data in enumerate(data[:]):  # Process all bills
    generate_bill_image(bill_data, i)
    print(f"Image {i} generated.")

print(f"✅ Generated {len(data)} bill images in '{OUTPUT_DIR}' (White Background Only)")

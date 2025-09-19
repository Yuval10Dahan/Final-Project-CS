from PIL import Image, ImageDraw, ImageFont
import os
import json

# Directory to save email images
image_output_folder = "email_images"
os.makedirs(image_output_folder, exist_ok=True)

# Load the JSON file containing the emails
json_file_path = os.path.join("json_files", "sensitive_bank_emails.json")
with open(json_file_path, "r") as file:
    emails = json.load(file)

# Font settings (adjust the font path for your system)
font_path = "arial.ttf"  # Change to a valid font file path on your system
header_font_size = 24
body_font_size = 16

try:
    header_font = ImageFont.truetype(font_path, header_font_size)
    body_font = ImageFont.truetype(font_path, body_font_size)
except:
    print("Font not found. Using default font.")
    header_font = ImageFont.load_default()
    body_font = ImageFont.load_default()


# Function to create an email image
def create_email_image(email, index):
    img_width, img_height = 850, 600
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)

    # Create a blank image
    img = Image.new("RGB", (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    # Draw header
    margin = 30
    draw.text((margin, margin), email["Bank"], fill=(0, 102, 204), font=header_font)

    # Draw email body
    body_text = email["Email Body"]
    y_offset = margin + 50  # Start below the header
    line_spacing = 25

    for line in body_text.split("\n"):
        draw.text((margin, y_offset), line, fill=text_color, font=body_font)
        y_offset += line_spacing

    # Save the image
    img.save(os.path.join(image_output_folder, f"email_{index}.png"))
    print(f"email_{index}.png saved")


# Generate images for the first 100 emails (adjust as needed)
for i, email in enumerate(emails[:]):
    create_email_image(email, i)

print(f"Email images saved in folder: {image_output_folder}")

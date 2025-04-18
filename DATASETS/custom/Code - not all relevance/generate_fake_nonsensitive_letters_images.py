from PIL import Image, ImageDraw, ImageFont
import json
import os

# Paths and constants
json_file_path = "json_files/nonsensitive_fake_letters.json"  # Path to the JSON file with letters
output_folder = "letter_images"  # Folder to save the images
os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists

# Font settings (adjust the font path as needed for your system)
font_path = "arial.ttf"  # Change this to a path of a valid font file on your system
font_size = 16
line_spacing = 24


# Function to create an image of the letter
def create_letter_image(letter_text, output_path):
    # Create a blank white image
    image_width, image_height = 800, 600
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Font not found: {font_path}. Using default font.")
        font = ImageFont.load_default()

    # Split text into lines
    lines = letter_text.split("\n")
    x, y = 10, 50  # Starting position for the text

    # Draw text line by line
    for line in lines:
        draw.text((x, y), line, font=font, fill="black")
        y += line_spacing

    # Save the image
    image.save(output_path)
    print(f"Saved letter image: {output_path}")


# Load the letters from the JSON file
with open(json_file_path, "r") as json_file:
    letters = json.load(json_file)

# Generate images for each letter
for i, letter in enumerate(letters):
    output_path = os.path.join(output_folder, f"letter_{i + 1}.png")
    create_letter_image(letter, output_path)

print(f"All letter images saved to {output_folder}")

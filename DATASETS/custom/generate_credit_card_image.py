from PIL import Image, ImageDraw, ImageFont
import json
import os

# Load the template
template_path = "creditCards/templates/black_card.png"  # Path to clean template
output_folder = "creditCards/output_cards/visa_black/"  # Folder to save the generated images

# Ensure the mastercard folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the JSON data
with open("json_files/cards.json", "r") as f:
    data = json.load(f)

# Font setup
font_path = "C:/Windows/Fonts/arial.TTF"  # Path to selected font
font_size = 35  # Adjust the font size as needed
font = ImageFont.truetype(font_path, font_size)

# Loop through each record and generate a card
for record in data:
    # Open a new copy of the template for each card
    template = Image.open(template_path)
    draw = ImageDraw.Draw(template)

    # Extract fields from JSON
    name = record["Name"]
    card_number = record["Credit_Card_Number"]
    formatted_number = f"{card_number[:4]}   {card_number[4:8]}   {card_number[8:12]}   {card_number[12:]}"  # Add spaces
    expiry_date = record["Expiry_Date"]

    # Add the details to the template
    draw.text((20, 200), formatted_number, fill="white", font=font)  # Card number with spaces
    draw.text((20, 35), name.upper(), fill="white", font=font)  # Name in uppercase
    draw.text((50, 290), expiry_date, fill="white", font=font)  # Expiry date

    # Save the image
    output_path = os.path.join(output_folder, f"{name.replace(' ', '_')}_card.png")
    template.save(output_path)
    print(f"Generated card for {name}: {output_path}")

from PIL import Image, ImageDraw, ImageFont
import json
import os

# Directory for JSON data
input_folder = "json_files"
json_file_path = os.path.join(input_folder, "phone_bills.json")

# Directory for generated images
output_folder = "phone_bill_images"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path to the uploaded logo
logo_path = "Logos/ATT_LOGO.png"

# Load data from JSON file
with open(json_file_path, "r") as json_file:
    phone_bills = json.load(json_file)

# Generate images
for i, bill in enumerate(phone_bills):
    customer_info = bill["Customer Information"]
    billing_period = bill["Billing Period"]
    charges_summary = bill["Charges Summary"]
    payment_info = bill["Payment Information"]

    # Create an image
    image = Image.new('RGB', (800, 1200), 'white')
    draw = ImageDraw.Draw(image)

    # Load fonts (ensure you have valid font paths)
    font = ImageFont.truetype("arial.ttf", 20)
    bold_font = ImageFont.truetype("arialbd.ttf", 22)  # Bold font for headlines
    title_font = ImageFont.truetype("arial.ttf", 30)  # Larger font for the title

    # Draw the title
    draw.text((330, 20), "Phone Bill", fill='black', font=title_font)
    y_position = 100  # Start below the title
    line_spacing = 30

    # Draw each section
    for section, content in {
        "Customer Information:": f"Name: {customer_info['Name']}\n"
                                 f"ID: {customer_info['ID']}\n"
                                 f"Address: {customer_info['Address']}\n"
                                 f"Phone Number: {customer_info['Phone Number']}",
        "Billing Period:": billing_period,
        "Charges Summary:": "\n".join([f"{key}: {value}" for key, value in charges_summary.items()]),
        "Payment Information:": "\n".join([f"{key}: {value}" for key, value in payment_info.items()])
    }.items():
        draw.text((20, y_position), section, fill='black', font=bold_font)  # Bold the section titles
        y_position += line_spacing
        for line in content.split('\n'):
            draw.text((40, y_position), line, fill='black', font=font)
            y_position += line_spacing
        y_position += 20  # Extra space between sections

    # Load and paste the logo
    logo = Image.open(logo_path).convert("RGBA")  # Convert to RGBA to handle transparency
    logo_resized = logo.resize((200, 100))  # Resize the logo to fit
    image.paste(logo_resized, (300, 1050), logo_resized)  # Use the logo as a mask for transparency

    # Save the image
    image_path = os.path.join(output_folder, f"phone_bill_{customer_info['Name']}.png")
    image.save(image_path)

    print(f"phone bill image number {i} saved to the folder")

print(f"Phone bill images saved to {output_folder}")

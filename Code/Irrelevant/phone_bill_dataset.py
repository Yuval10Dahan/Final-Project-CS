from faker import Faker
from components import credit_card
from components import ID
from components import driver_licence
from components import bank_account
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize Faker and components
fake = Faker()
card = credit_card
driver = driver_licence
Id = ID
bank = bank_account

# Directory for generated images
output_folder = "phone_bill_images"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path to the uploaded logo
logo_path = "Logos/ATT_LOGO.png"

# Generate data and create images
for i in range(50000):  # Adjust the range for more bills
    name = fake.name()
    address = fake.address()
    phone_number = fake.basic_phone_number()
    id_number = Id.generate_fake_id()
    date = fake.date()
    total_amount_due = fake.random_number(digits=2, fix_len=True) + fake.random_number(digits=2) / 100
    monthly_service_charge = total_amount_due * 0.4
    call_charges = total_amount_due * 0.3
    sms_charges = total_amount_due * 0.2
    data_usage_charges = total_amount_due * 0.1

    # Generate start and end dates for the billing period
    start_date = fake.date_between(start_date='-1y', end_date='-1d')
    end_date = fake.date_between(start_date=start_date, end_date='today')

    # Create phone bill data
    phone_bill = {
        "Customer Information:": f"Name: {name}\nID: {id_number}\nAddress: {address}\nPhone Number: {phone_number}",
        "Billing Period:": f"{start_date.isoformat()} – {end_date.isoformat()}",
        "Charges Summary:": f"Monthly Service Charge: ${monthly_service_charge:.2f}\n"
                            f"Call Charges: ${call_charges:.2f}\n"
                            f"SMS Charges: ${sms_charges:.2f}\n"
                            f"Data Usage: ${data_usage_charges:.2f}\n"
                            f"Total: ${total_amount_due:.2f}",
        "Payment Information:": f"Total Amount Due: ${total_amount_due:.2f}\n"
                                f"Due Date: {fake.date_between('today', '+30d').isoformat()}\n"
                                f"Bank Account Number: {bank.generate_fake_bank_account()}"
    }

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
    for section, content in phone_bill.items():
        draw.text((20, y_position), section, fill='black', font=bold_font)  # Bold the section titles
        y_position += line_spacing
        for line in content.split('\n'):
            draw.text((40, y_position), line, fill='black', font=font)
            y_position += line_spacing
        y_position += 20  # Extra space between sections

    # Load and paste the logo
    logo = Image.open(logo_path).convert("RGBA")  # Convert to RGBA to handle transparency
    logo_resized = logo.resize((200, 100))  # Resize the logo to fit
    image.paste(logo_resized, (300, 800), logo_resized)  # Use the logo as a mask for transparency

    # Save the image
    image_path = os.path.join(output_folder, f"phone_bill_{name}.png")
    image.save(image_path)

print(f"Phone bill images saved to {output_folder}")

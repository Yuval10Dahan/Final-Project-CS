import json
import random
import os
from faker import Faker

fake = Faker()

# Directory where the JSON file will be saved
output_folder = "json_files"
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "sensitive_bank_emails.json")

bank_names = ["JPMorgan Chase Bank", "Bank of America", "Wells Fargo Bank", "Citibank", "U.S. Bank"]


def generate_sensitive_email():
    client_name = fake.name()
    credit_card_number = "".join([str(random.randint(0, 9)) for _ in range(16)])
    pin_code = "".join([str(random.randint(0, 9)) for _ in range(4)])
    bank_name = random.choice(bank_names)

    # Generate the full email body
    email_body = (
        f"Subject: Your New Credit Card PIN\n\n"
        f"Dear {client_name},\n\n"
        f"Thank you for choosing our bank. \nWe are pleased to inform you that your new credit card "
        f"ending in {credit_card_number[-4:]} has been issued successfully.\n\n"
        f"For your security, your new PIN code is: {pin_code}. \nPlease keep this PIN confidential "
        f"and do not share it with anyone. \n"
        f"If you have any questions, please contact our customer service team at the application.\n\n"
        f"Best regards,\n{bank_name}"
    )

    return {
        "Client Name": client_name,
        "Credit Card Number": credit_card_number,
        "PIN": pin_code,
        "Email Body": email_body,
        "Bank": bank_name
    }


# Generate 50,000 email messages
data = [generate_sensitive_email() for _ in range(30000)]

# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with sensitive bank emails saved as {file_path}.")

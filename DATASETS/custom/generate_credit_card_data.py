from faker import Faker
from components import credit_card
import json
import os

# variables
fake = Faker()
card = credit_card

# parameters for the credit card number
prefix = 4580  # Visa prefix
length = 16  # Typical length for Visa cards

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "cards.json")

# generate data
data = []

for _ in range(1):  # Adjust the range for more data
    entry = {
        "Name": fake.name(),
        "Credit_Card_Number": credit_card.generate_card_number(prefix, length),
        "Expiry_Date": fake.credit_card_expire()
    }
    data.append(entry)

# Save to a JSON file
with open("json_files/cards.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

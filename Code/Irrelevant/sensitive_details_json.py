from faker import Faker
from components import credit_card
from components import ID
from components import driver_licence
from components import bank_account
import json
import os

# variables
fake = Faker()
card = credit_card
driver = driver_licence
Id = ID
bank = bank_account

# parameters for the credit card number
prefix = 4580  # Visa prefix
length = 16  # Typical length for Visa cards

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "PII.json")

# generate data
data = []

for _ in range(100000):  # Adjust the range for more data
    entry = {
        "Name": fake.name(),
        # "address": fake.address(),
        "Social Security Number": fake.ssn(),
        "Credit card number": card.generate_card_number(prefix, length),
        "Passport number": fake.passport_number(),
        "ID": ID.generate_fake_id(),
        "Bank account number": bank.generate_fake_bank_account()
    }
    data.append(entry)

# Save to a JSON file
with open("PII.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

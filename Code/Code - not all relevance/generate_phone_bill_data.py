from faker import Faker
from components import credit_card
from components import ID
from components import driver_licence
from components import bank_account
import json
import os

# Initialize Faker and components
fake = Faker()
card = credit_card
driver = driver_licence
Id = ID
bank = bank_account

# Directory for JSON data
output_folder = "json_files"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path for the JSON file
json_file_path = os.path.join(output_folder, "phone_bills.json")

# Generate data
data = []
for _ in range(2):  # Adjust the range for more bills
    name = fake.name()
    address = fake.address()
    phone_number = fake.basic_phone_number()
    id_number = Id.generate_fake_id()
    total_amount_due = fake.random_number(digits=2, fix_len=True) + fake.random_number(digits=2) / 100
    monthly_service_charge = total_amount_due * 0.4
    call_charges = total_amount_due * 0.3
    sms_charges = total_amount_due * 0.2
    data_usage_charges = total_amount_due * 0.1

    # Generate start and end dates for the billing period
    start_date = fake.date_between(start_date='-1y', end_date='-1d')
    end_date = fake.date_between(start_date=start_date, end_date='today')

    # Add data to list
    data.append({
        "Customer Information": {
            "Name": name,
            "ID": id_number,
            "Address": address,
            "Phone Number": phone_number
        },
        "Billing Period": f"{start_date.isoformat()} – {end_date.isoformat()}",
        "Charges Summary": {
            "Monthly Service Charge": f"${monthly_service_charge:.2f}",
            "Call Charges": f"${call_charges:.2f}",
            "SMS Charges": f"${sms_charges:.2f}",
            "Data Usage": f"${data_usage_charges:.2f}",
            "Total": f"${total_amount_due:.2f}"
        },
        "Payment Information": {
            "Total Amount Due": f"${total_amount_due:.2f}",
            "Due Date": fake.date_between('today', '+30d').isoformat(),
            "Bank Account Number": bank.generate_fake_bank_account()
        }
    })

# Save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"Phone bill data saved to {json_file_path}")

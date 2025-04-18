from faker import Faker
from components import credit_card, ID, driver_licence, bank_account
import json
import os
import random

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
json_file_path = os.path.join(output_folder, "sensitive_phone_bills.json")

# Carrier names for diversity
carriers = [
    "Verizon Wireless", "AT&T Mobility", "T-Mobile", "Sprint", "Vodafone",
    "Orange Telecom", "Bell Canada", "Telstra", "Rogers Wireless", "Deutsche Telekom",
    "O2 UK", "Virgin Mobile", "Boost Mobile", "Metro by T-Mobile", "US Cellular",
    "Claro", "Movistar", "Telkom", "Globe Telecom", "Reliance Jio",
    "Airtel", "MTN Group", "China Mobile", "China Unicom", "Singtel",
    "KDDI Japan", "NTT Docomo", "Swisscom", "Telefónica", "BT Mobile",
    "Wind Tre", "Optus", "Telenor", "SK Telecom", "KT Corporation",
    "Mobily", "Etisalat", "STC Saudi Arabia", "Zain", "Jazz Telecom"
]


# Currency formats
currencies = [
    "$", "USD", "CAD", "€", "EUR", "GBP", "£", "¥", "JPY", "₩", "KRW",
    "₹", "INR", "₱", "PHP", "CHF", "AUD", "NZD", "SAR", "AED", "BRL",
    "MXN", "RUB", "HKD", "SGD", "NOK", "SEK", "DKK", "ZAR", "PLN", "CZK"
]


# Different billing period formats
billing_period_templates = [
    "{start} to {end}",
    "{start} - {end}",
    "Billing cycle: {start} – {end}",
    "Period: {start} to {end}",
    "Billing statement for {start} through {end}",
    "Service charges for {start} until {end}",
    "Usage period: {start} - {end}",
    "From {start} until {end} (billing cycle)",
    "Charges applicable from {start} to {end}",
    "This bill covers the period: {start} – {end}",
    "Statement period: {start} to {end}",
    "Account activity from {start} to {end}",
    "Billing summary for {start} through {end}",
    "Cycle dates: {start} - {end}",
    "Chargeable period: {start} to {end}"
]


# Different due date phrasings
due_date_templates = [
    "Due by {date}",
    "Payment deadline: {date}",
    "Final payment date: {date}",
    "Must be paid before {date}",
    "Last day to pay: {date}",
    "Settlement required before {date}",
    "Please clear your dues by {date}",
    "Amount due no later than {date}",
    "Payment cutoff date: {date}",
    "Final due date: {date}",
    "Please ensure payment by {date}",
    "Outstanding balance must be paid before {date}",
    "Final notice: Payment due by {date}",
    "Ensure payment completion by {date}",
    "Balance to be settled by {date}"
]


# Different invoice formats
invoice_formats = [
    "INV-{num}",
    "BILL-{num}",
    "INVOICE-{num}",
    "Statement #{num}",
    "Ref No: {num}",
    "Invoice ID: {num}",
    "Billing Ref: {num}",
    "Charge Slip: {num}",
    "Payment Notice: {num}",
    "Account Statement {num}",
    "Customer Invoice #{num}",
    "Service Bill #{num}",
    "Order Invoice: {num}",
    "Transaction ID: {num}",
    "Billing Confirmation No: {num}",
    "Document Ref No: {num}",
    "Bill Summary: {num}",
    "Statement Reference: {num}",
    "Usage Invoice: {num}",
    "Monthly Statement ID: {num}"
]

# Function to add typos for data augmentation
def introduce_typos(text):
    if random.random() < 0.15:  # 15% chance of a typo
        index = random.randint(0, len(text) - 2)
        text = text[:index] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[index + 1:]
    return text

# Function to introduce random number formatting
def format_currency(amount):
    currency_symbol = random.choice(currencies)
    return f"{currency_symbol}{amount:,.2f}" if random.random() < 0.7 else f"{amount:,.2f} {currency_symbol}"

# Function to generate a fake phone bill
def generate_fake_phone_bill():
    name = fake.name()
    address = fake.address()
    phone_number = fake.basic_phone_number()
    id_number = Id.generate_fake_id()
    carrier = random.choice(carriers)

    # Invoice number generation
    invoice_number = random.choice(invoice_formats).format(num=random.randint(100000, 999999))

    # Billing period
    start_date = fake.date_between(start_date='-1y', end_date='-1d')
    end_date = fake.date_between(start_date=start_date, end_date='today')
    billing_period = random.choice(billing_period_templates).format(
        start=start_date.isoformat(),
        end=end_date.isoformat()
    )

    # Charges calculation
    total_amount_due = fake.random_number(digits=2, fix_len=True) + fake.random_number(digits=2) / 100
    monthly_service_charge = total_amount_due * random.uniform(0.35, 0.45)
    call_charges = total_amount_due * random.uniform(0.25, 0.35)
    sms_charges = total_amount_due * random.uniform(0.15, 0.25)
    data_usage_charges = total_amount_due * random.uniform(0.05, 0.15)
    taxes = total_amount_due * 0.07  # 7% tax
    late_fee = total_amount_due * 0.05 if random.random() < 0.2 else 0  # 20% chance of late fee
    discount = total_amount_due * 0.1 if random.random() < 0.3 else 0  # 30% chance of discount

    # Adjust total amount
    final_total = (total_amount_due + taxes + late_fee) - discount

    # Due date
    due_date = fake.date_between('today', '+30d')
    formatted_due_date = random.choice(due_date_templates).format(date=due_date.isoformat())

    # Bank details
    bank_name = fake.company()
    bank_account_number = bank.generate_fake_bank_account()
    payment_status = random.choice(["Paid", "Pending", "Overdue"])

    return {
        "Invoice Information": {
            "Carrier": carrier,
            "Invoice Number": introduce_typos(invoice_number),
            "Billing Period": introduce_typos(billing_period)
        },
        "Customer Information": {
            "Name": introduce_typos(name),
            "ID": id_number,
            "Address": introduce_typos(address),
            "Phone Number": introduce_typos(phone_number)
        },
        "Charges Summary": {
            "Monthly Service Charge": format_currency(monthly_service_charge),
            "Call Charges": format_currency(call_charges),
            "SMS Charges": format_currency(sms_charges),
            "Data Usage": format_currency(data_usage_charges),
            "Taxes & Fees": format_currency(taxes),
            "Late Fee": format_currency(late_fee) if late_fee else "N/A",
            "Discount": format_currency(-discount) if discount else "N/A",
            "Total Due": format_currency(final_total)
        },
        "Payment Information": {
            "Total Amount Due": format_currency(final_total),
            "Due Date": formatted_due_date,
            "Payment Status": payment_status,
            "Bank Name": bank_name,
            "Bank Account Number": bank_account_number
        }
    }


# Generate multiple fake phone bills
num_bills = 50000  # Change to desired number
data = [generate_fake_phone_bill() for _ in range(num_bills)]

# Save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"📜 {num_bills} fake phone bills saved to {json_file_path}")

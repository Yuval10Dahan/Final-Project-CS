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

# Bank names and subject variations
bank_names = [
    "JPMorgan Chase Bank", "Bank of America", "Wells Fargo Bank",
    "Citibank", "U.S. Bank", "Capital One", "PNC Bank", "TD Bank",
    "Goldman Sachs", "Morgan Stanley", "HSBC", "Barclays", "Lloyds Bank",
    "Deutsche Bank", "BNP Paribas", "Santander Bank", "UBS", "Royal Bank of Canada",
    "Scotiabank", "Bank of Montreal", "ICBC", "HDFC Bank", "DBS Bank", "Commonwealth Bank",
    "Westpac", "ANZ Bank", "First Abu Dhabi Bank", "Banco do Brasil", "Itaú Unibanco"
]

email_subjects = [
    "Important: Your New Credit Card PIN",
    "Your Secure PIN Code for Card Activation",
    "Notice: Credit Card Issued & PIN Confirmation",
    "Bank Alert: Your New Credit Card & PIN Details",
    "Security Notice: Your New PIN Has Been Generated",
    "Urgent: Your Credit Card is Ready for Use",
    "Action Required: Activate Your New Credit Card",
    "Your Credit Card Has Been Approved - PIN Inside",
    "Final Step: Retrieve Your Secure PIN Code",
    "Welcome! Your New Card & PIN Are Ready",
    "Secure Your Account: Your PIN Code Inside",
    "Your New Card Is On Its Way – Activation Details",
    "Credit Card Update: New PIN Code Assigned",
    "Your New Bank Card Has Been Activated",
    "PIN Reminder: Keep Your Code Confidential",
    "Bank Notice: Important PIN Security Update",
    "Fraud Prevention Alert: Protect Your PIN",
    "New Account Setup: Here’s Your Secure PIN",
    "Your Bank Account Has Been Updated – PIN Details",
    "Your Debit Card PIN Code Has Been Changed",
    "Credit Card Replacement: New PIN Code Inside",
    "PIN Confirmation: Verify Your Secure Code",
    "Important: Do Not Share Your PIN Code",
    "Your Digital Wallet Is Now Linked to Your Card",
    "Account Update: Secure PIN Confirmation Needed",
    "Reminder: Do Not Share Your Secure PIN",
    "New Banking Regulations: PIN Change Required",
    "PIN Reset Request: Verify Your Identity",
    "New Card Issued – Please Secure Your PIN",
    "Your New Credit Card Is Ready – PIN Code Enclosed",
    "Security Advisory: PIN Change Recommended",
    "Important Security Notice: New PIN Assigned"
]


# Sentence structures for variation
email_templates = [
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your credit card ending in {card_end} has been issued. Your PIN code is {pin}. "
        f"Keep it confidential. \n\nBest,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Hello {name},\n\n"
        f"Your new {bank} card (****{card_end}) is active. "
        f"Use this PIN: {pin}. Never share it.\n\nSincerely,\n{bank}."
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your {bank} credit card with last digits {card_end} has been issued. "
        f"To activate, use PIN: {pin}. Keep it private.\n\nRegards,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"We are pleased to inform you that your {bank} credit card (****{card_end}) "
        f"has been successfully issued. \nYour secure PIN code is: {pin}. \n"
        f"Please do not share this information with anyone.\n\nBest,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your new credit card from {bank} has been dispatched. "
        f"To activate your card ending in {card_end}, please use PIN: {pin}. \n"
        f"Do not share your PIN with anyone for security reasons.\n\nThank you,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your {bank} account has been updated with a new card (****{card_end}). "
        f"For your security, your PIN is {pin}. Keep this number private and do not disclose it.\n\n"
        f"For any assistance, please contact our support team.\n\nSincerely,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"We have successfully generated a new PIN for your {bank} credit card (****{card_end}). "
        f"Your temporary PIN is {pin}. \n"
        f"You may change this PIN at any time through your banking portal.\n\n"
        f"Stay secure,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Hello {name},\n\n"
        f"Your card ending in {card_end} is now activated. Use PIN {pin} for transactions. \n"
        f"Always keep your PIN confidential to avoid unauthorized access.\n\nThank you,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your new {bank} credit card (last four digits: {card_end}) has been issued successfully. "
        f"Your assigned PIN is {pin}. \n"
        f"For your safety, never share this PIN with anyone.\n\nBest regards,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Welcome to {bank}! Your new card (****{card_end}) is ready to use. "
        f"Please use this PIN: {pin} to access your account.\n\n"
        f"If you have any concerns, contact us immediately.\n\nSincerely,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Attention {name},\n\n"
        f"Your credit card with last four digits {card_end} has been successfully activated. "
        f"Your unique PIN code is {pin}. Keep it safe and secure.\n\n"
        f"For security-related queries, contact {bank} support.\n\nRegards,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"We are excited to inform you that your {bank} card (****{card_end}) is now active. "
        f"Your security PIN is {pin}. Do not disclose it under any circumstances.\n\n"
        f"Thank you for choosing {bank}.\n\nBest,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Hello {name},\n\n"
        f"Your new {bank} card ending in {card_end} is now linked to your account. "
        f"Use PIN {pin} for secure transactions. \n"
        f"To protect your privacy, never share your PIN with anyone.\n\nBest,\n{bank}"
    ),
    lambda name, card_end, pin, bank: (
        f"Dear {name},\n\n"
        f"Your new {bank} credit card ending in {card_end} has been issued. "
        f"Your assigned PIN is {pin}. \n"
        f"Make sure to keep this information confidential to avoid unauthorized transactions.\n\n"
        f"Thank you,\n{bank}"
    )
]


# Function to create variations in sensitive emails
def generate_sensitive_email():
    client_name = fake.name()
    credit_card_number = "".join([str(random.randint(0, 9)) for _ in range(16)])
    card_end = credit_card_number[-4:]
    pin_code = "".join([str(random.randint(0, 9)) for _ in range(4)])
    cvv = "".join([str(random.randint(0, 9)) for _ in range(3)])  # Additional security detail
    expiry_date = f"{random.randint(1, 12)}/{random.randint(24, 30)}"
    bank_name = random.choice(bank_names)
    subject = random.choice(email_subjects)
    template = random.choice(email_templates)

    # Introduce occasional typos or spacing variations for augmentation
    if random.random() < 0.1:  # 10% chance to introduce variation
        pin_code = pin_code[:2] + " " + pin_code[2:]  # Add a space inside the PIN
        card_end = "*".join(card_end)  # Make the card ending appear obfuscated
        cvv = cvv[0] + " " + cvv[1:]  # Add spacing to CVV for randomness

    email_body = template(client_name, card_end, pin_code, bank_name)

    return {
        "Client Name": client_name,
        "Credit Card Number": credit_card_number,
        "Card Ending": card_end,
        "PIN": pin_code,
        "CVV": cvv,
        "Expiry Date": expiry_date,
        "Email Subject": subject,
        "Email Body": email_body,
        "Bank": bank_name
    }


# Generate dataset with variability
data = [generate_sensitive_email() for _ in range(30000)]


# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with enhanced sensitive bank emails saved as {file_path}.")

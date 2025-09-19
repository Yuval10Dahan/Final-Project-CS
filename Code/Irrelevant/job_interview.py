from faker import Faker
import random
import os
import json

# Initialize Faker
fake = Faker()

# Directory for JSON data
output_folder = "json_files"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path for the JSON file
json_file_path = os.path.join(output_folder, "job_interview_invitations.json")

# List of companies
companies = [
    "Adobe",
    "Akamai Technologies",
    "Amdocs",
    "Ansys",
    "Atlassian",
    "Broadridge Financial",
    "Citrix Systems",
    "Datadog",
    "DocuSign",
    "EPAM Systems",
    "Fleetcor",
    "Gen Digital",
    "Global Payments",
    "Google",
    "Hubspot",
    "IBM",
    "Kyndryl",
    "Microsoft",
    "NetApp",
    "Okta",
    "Oracle",
    "Paycom",
    "PayPal",
    "RingCentral",
    "SAP",
    "SS&C Technologies",
    "Solutions",
    "Splunk",
    "Twilio",
    "Unity Technologies",
    "Veeva Systems",
    "Zoom"
]

# Function to generate a fake job interview invitation


def generate_job_invitations(number_of_invitations):
    data = []

    for _ in range(number_of_invitations):
        # Generate fake name and company details
        first_name = fake.first_name()
        last_name = fake.last_name()
        sender_first_name = fake.first_name()
        sender_last_name = fake.last_name()
        company_name = random.choice(companies)
        # Generate a fake email that includes the company name
        company_email = f"{sender_first_name.lower()}.{sender_last_name.lower()}@{company_name.replace(' ', '').lower()}.com"
        company_address = fake.address().replace("\n", ", ")  # Replace newline with comma for a cleaner address
        company_phone = fake.basic_phone_number()

        # Generate a random round hour for the time
        time_hour = random.choice(range(9, 18))  # Office hours from 9 AM to 5 PM
        interview_time = f"{time_hour:02d}:00"

        # Construct the invitation letter
        letter = f"""
        Dear {first_name} {last_name},

        We are excited to invite you to a job interview for a position at {company_name}. 

        Interview Details:
        - Date: {fake.date_this_month(before_today=False, after_today=True).strftime('%A, %d %B %Y')}
        - Time: {interview_time} (Local Time)
        - Location: {company_address}

        This is an excellent opportunity to learn more about our company, {company_name},
        and discuss how you can contribute to our team.

        If you have any questions or need to reschedule, please contact me at:
        Email: {company_email}
        Phone: {company_phone}

        We look forward to meeting you!

        Best regards,
        {sender_first_name} {sender_last_name}
        Human Resources
        {company_name}
        """

        data.append(letter.strip())

    return data


# Generate and save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(generate_job_invitations(1), json_file, indent=4)

print(f"Job interview invitations saved to {json_file_path}")

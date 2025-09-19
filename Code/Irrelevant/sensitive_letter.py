from faker import Faker
import random
import os
import json
from components import ID
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Directory for JSON data
output_folder = "json_files"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path for the JSON file
json_file_path = os.path.join(output_folder, "sensitive_fake_letters.json")

# List of non-sensitive topics
topics = [
    "your medical appointment.",
    "a medical appointment you have."
]

# Templates for letter content
templates = [
    "We wanted to remind you about {topic}",
    "Here's a quick reminder about {topic}",
    "This is a friendly reminder about {topic}",
]


# Function to generate a fake letter
def generate_sensitive_fake_letter(number_of_letters):
    data = []

    for _ in range(number_of_letters):
        # Generate fake personal and sensitive information
        first_name = fake.first_name()
        last_name = fake.last_name()
        id_number = ID.generate_fake_id()
        ssn = fake.ssn()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d")
        phone_number = fake.basic_phone_number()
        medical_records = (f"Blood type: {random.choice(['A', 'B', 'AB', 'O'])},"
                           f" Known allergies: {random.choice(['None', 'Peanuts', 'Pollen', 'Dust'])}")

        # Generate up-to-date appointment date and time
        appointment_date = (datetime.now() + timedelta(days=random.randint(0, 7))).strftime("%Y-%m-%d")
        appointment_time = (datetime.now() + timedelta(hours=random.randint(0, 23))).strftime("%H:%M")

        # Pick a random non-sensitive topic
        topic = random.choice(topics)
        letter_body = random.choice(templates).format(topic=topic)

        # Construct the letter
        letter = f"""
        Dear {first_name} {last_name},

        We hope this letter finds you well.
        {letter_body}
        
        Appointment details:
        Date: {appointment_date}
        Time: {appointment_time}

        Inform us in case of change in your personal details:
        - ID: {id_number}
        - Social Security Number (SSN): {ssn}
        - Date of Birth: {dob}
        - Phone Number: {phone_number}
        - Medical Records: 
            {medical_records}

        For any questions or further assistance, 
        feel free to contact us at our mail: {fake.company_email()}.

        Best regards,
        {fake.name()}
        """

        data.append(letter.strip())  # Strip extra spaces for cleaner output

    return data


# Save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(generate_sensitive_fake_letter(1), json_file, indent=4)

print(f"fake letters data saved to {json_file_path}")

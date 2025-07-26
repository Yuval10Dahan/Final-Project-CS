from faker import Faker
import random
import os
import json
from datetime import datetime, timedelta
import re

# Initialize Faker
fake = Faker()

# Directory for JSON data
output_folder = "json_files"
os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

# Path for the JSON file
json_file_path = os.path.join(output_folder, "sensitive_medical_letters.json")

# List of realistic medical topics
topics = [
    "your upcoming medical appointment.",
    "a scheduled consultation with your doctor.",
    "your check-up appointment at our clinic.",
    "a health assessment you have with our team.",
    "a diagnostic test that has been scheduled.",
    "your upcoming visit to our hospital.",
    "a follow-up consultation regarding your recent test results.",
    "a vaccination appointment scheduled for you.",
    "a routine health screening you are due for.",
    "your annual physical examination appointment.",
    "a dental check-up that has been booked for you.",
    "an eye examination appointment at our optometry clinic.",
    "a therapy session scheduled with your specialist.",
    "a cardiology check-up with our expert team.",
    "your upcoming physiotherapy session.",
    "a mental health counseling appointment with our therapist.",
    "your upcoming telehealth consultation.",
    "a lab test that has been scheduled for your medical review.",
    "your diagnostic imaging (MRI/CT scan) appointment.",
    "an outpatient procedure planned for you.",
    "your nutritionist consultation regarding your diet plan.",
    "a post-operative follow-up with your surgeon.",
    "your upcoming appointment for a second medical opinion.",
    "a pediatric appointment scheduled for your child.",
    "your dermatology appointment to discuss your skin condition.",
    "your endocrinology consultation regarding your hormone levels.",
    "a scheduled allergy test to determine your sensitivities.",
    "your pre-employment medical examination.",
    "a fitness assessment scheduled at our health center.",
    "your hearing test appointment at our audiology clinic.",
    "your sports medicine consultation for injury prevention.",
    "an upcoming gynecology appointment for routine screening.",
    "a pain management consultation with our specialist.",
    "your general wellness check-up to maintain good health.",
    "a rehabilitation session planned for your recovery.",
    "an orthopedic appointment regarding joint pain concerns.",
    "a home healthcare visit scheduled with our nurse.",
    "your upcoming blood donation appointment at the center.",
    "a smoking cessation consultation to help you quit smoking.",
    "a weight management consultation to assist your health journey."
]


# Templates for letter content
templates = [
    "We wanted to remind you about {topic}",
    "This is a friendly reminder regarding {topic}",
    "Here's an important update about {topic}",
    "We hope this message finds you well. A quick note regarding {topic}",
    "Please take note of the details below for {topic}",
    "This is to confirm your upcoming medical visit. {topic}",
    "Your scheduled appointment details are included below for {topic}",
    "As a courtesy, we would like to remind you about {topic}",
    "We are reaching out to inform you about {topic}",
    "Just a quick reminder about {topic}. Please review the details carefully.",
    "This message serves as a reminder for {topic}. Let us know if you have any questions.",
    "Your upcoming appointment is approaching. Here’s what you need to know about {topic}.",
    "Please find the details for {topic} below.",
    "We have scheduled your next session for {topic}. Please confirm your attendance.",
    "We wanted to follow up on {topic}. Kindly review your appointment details.",
    "Your scheduled visit is coming up soon. Here’s the information for {topic}.",
    "A gentle reminder that {topic} is scheduled. Please mark your calendar.",
    "Just letting you know that your scheduled appointment for {topic} is approaching.",
    "Don't forget about {topic}. Please let us know if you need to make any changes.",
    "Your appointment details for {topic} are now available. Please review the information carefully.",
    "We noticed you have an upcoming session for {topic}. Reach out if you need assistance.",
    "Your upcoming appointment for {topic} has been confirmed. Please arrive on time.",
    "To ensure everything goes smoothly, please review the information for {topic}.",
    "Here’s a quick heads-up about {topic}. Let us know if you need to reschedule.",
    "We would like to remind you about your scheduled appointment for {topic}.",
    "Mark your calendar for {topic}. Please arrive at the designated time.",
    "Your scheduled time for {topic} is approaching. Please confirm your availability.",
    "This email serves as confirmation of your appointment for {topic}.",
    "Your upcoming consultation for {topic} has been successfully scheduled.",
    "Just a quick update regarding {topic}. Let us know if anything changes.",
    "We appreciate your time and want to remind you about {topic}.",
    "A brief reminder that {topic} is set for the upcoming days.",
    "Your upcoming session for {topic} is on the calendar. Please be on time.",
    "A kind reminder that {topic} is scheduled. We look forward to seeing you.",
    "Please note that {topic} is confirmed. If you have any concerns, let us know.",
    "Here’s a reminder regarding {topic}. Kindly review the appointment details.",
    "Your upcoming consultation for {topic} is important. Let us know if you have questions.",
    "We are sending you this message to confirm {topic}. Please check your schedule.",
    "A notification to let you know that {topic} is scheduled.",
    "An important notice regarding {topic}. Please review the details below.",
    "Don't forget about {topic}! We have all the details you need right here."
]


# Randomized closing statements
closing_statements = [
    "If you have any questions, feel free to contact our office.",
    "For further assistance, reach out to our support team.",
    "Please confirm your attendance at your earliest convenience.",
    "If you need to reschedule, let us know as soon as possible.",
    "We appreciate your time and look forward to seeing you.",
    "Should you have any concerns, don’t hesitate to reach out.",
    "If you require any modifications to your appointment, please inform us.",
    "Feel free to get in touch if you have any doubts or need assistance.",
    "We are here to assist you. Let us know if you need any clarification.",
    "If this appointment is no longer convenient, please let us know promptly.",
    "To ensure a smooth experience, please review the provided details carefully.",
    "Please don't hesitate to ask if you require any additional information.",
    "If you need special arrangements, contact us ahead of time.",
    "Your prompt confirmation will help us serve you better.",
    "Let us know at your earliest convenience if you need any changes.",
    "We appreciate your cooperation and are happy to assist in any way.",
    "For urgent concerns, please call our office directly.",
    "We value your time and hope to make your visit as smooth as possible.",
    "If there are any issues, feel free to reach out for support.",
    "Our team is available to address any inquiries you may have.",
    "Your confirmation would be greatly appreciated.",
    "Should you have any questions, we’re happy to assist.",
    "We look forward to seeing you at the scheduled time.",
    "Your satisfaction is our priority. Let us know how we can help.",
    "If anything is unclear, don't hesitate to contact us.",
    "We’re happy to help if you need any changes to your appointment.",
    "Please keep this appointment in mind and arrive on time.",
    "We are committed to providing excellent service. Reach out anytime.",
    "If you experience any scheduling conflicts, let us know as soon as possible.",
    "For any last-minute changes, please notify us at your earliest convenience.",
    "Please ensure all necessary documents are ready for your appointment.",
    "We appreciate your cooperation and look forward to assisting you.",
    "Looking forward to your visit! Feel free to reach out if needed.",
    "We want to make this process as seamless as possible for you.",
    "If anything comes up, you can always reschedule with us.",
    "We thank you for your attention and hope to provide the best experience possible.",
    "We are always here to provide any support you may need.",
    "If you need assistance before your visit, don’t hesitate to get in touch.",
    "Have a great day, and we look forward to serving you soon."
]


# Signature styles
signatures = [
    "Best regards,\n{doctor_name}",
    "Sincerely,\n{doctor_name}",
    "Warm wishes,\n{doctor_name}",
    "Respectfully,\n{doctor_name}",
    "Thank you,\n{doctor_name}",
    "Kind regards,\n{doctor_name}",
    "With appreciation,\n{doctor_name}",
    "With warm regards,\n{doctor_name}",
    "Yours sincerely,\n{doctor_name}",
    "Cordially,\n{doctor_name}",
    "With best wishes,\n{doctor_name}",
    "Gratefully,\n{doctor_name}",
    "Yours truly,\n{doctor_name}",
    "Many thanks,\n{doctor_name}",
    "With my best,\n{doctor_name}",
    "Best wishes,\n{doctor_name}",
    "With kindest regards,\n{doctor_name}",
    "Looking forward to seeing you,\n{doctor_name}",
    "Hoping to assist you soon,\n{doctor_name}",
    "Wishing you good health,\n{doctor_name}",
    "In good faith,\n{doctor_name}",
    "To your well-being,\n{doctor_name}",
    "With heartfelt thanks,\n{doctor_name}",
    "Yours in service,\n{doctor_name}",
    "With dedication,\n{doctor_name}",
    "In gratitude,\n{doctor_name}",
    "Thank you for your trust,\n{doctor_name}",
    "Wishing you the best,\n{doctor_name}",
    "Appreciatively,\n{doctor_name}",
    "With sincere regards,\n{doctor_name}",
    "We appreciate your time,\n{doctor_name}",
    "Take care,\n{doctor_name}",
    "Wishing you a speedy recovery,\n{doctor_name}",
    "Hope to see you soon,\n{doctor_name}",
    "Yours in good health,\n{doctor_name}",
    "On behalf of our team,\n{doctor_name}",
    "Stay well,\n{doctor_name}",
    "With utmost respect,\n{doctor_name}",
    "Committed to your care,\n{doctor_name}",
    "Always here to help,\n{doctor_name}"
]



# Function to introduce random typos for augmentation
def introduce_typos(text):
    if random.random() < 0.2:  # 20% chance of a typo
        typo_index = random.randint(0, len(text) - 2)
        text = text[:typo_index] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[typo_index + 1:]
    return text


# Function to introduce random formatting inconsistencies
def introduce_formatting_variations(text):
    if random.random() < 0.3:  # 30% chance to modify text
        text = re.sub(r'\.', '...', text)  # Replace periods with ellipses randomly
    if random.random() < 0.2:
        text = re.sub(r'(\w)(,)', r'\1 ,', text)  # Add space before commas
    if random.random() < 0.2:
        text = text.replace("  ", "   ")  # Randomize spacing
    return text


# Function to generate a fake letter
def generate_sensitive_fake_letter(number_of_letters):
    data = []

    for _ in range(number_of_letters):
        # Generate fake personal and sensitive information
        first_name = fake.first_name()
        last_name = fake.last_name()
        id_number = fake.random_number(digits=9, fix_len=True)  # Random 9-digit ID
        ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        dob = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d")
        phone_number = fake.phone_number()
        medical_records = f"Blood type: {random.choice(['A', 'B', 'AB', 'O'])}, Known allergies: {random.choice(['None', 'Peanuts', 'Pollen', 'Dust'])}"

        # Generate up-to-date appointment date and time
        appointment_date = (datetime.now() + timedelta(days=random.randint(0, 14))).strftime("%Y-%m-%d")
        appointment_time = (datetime.now() + timedelta(hours=random.randint(0, 23))).strftime("%H:%M")

        # Pick a random non-sensitive topic
        topic = random.choice(topics)
        letter_body = random.choice(templates).format(topic=topic)

        # Pick a closing statement and signature
        closing = random.choice(closing_statements)
        doctor_name = fake.name()
        signature = random.choice(signatures).format(doctor_name=doctor_name)

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

        {closing}
        You can reach us at {fake.company_email()}.

        {signature}
        """

        # Introduce typos and formatting variations for augmentation
        letter = introduce_typos(letter)
        letter = introduce_formatting_variations(letter)

        data.append(letter.strip())  # Strip extra spaces for cleaner output

    return data


# Save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(generate_sensitive_fake_letter(50000), json_file, indent=4)

print(f"Fake letters data saved to {json_file_path}")

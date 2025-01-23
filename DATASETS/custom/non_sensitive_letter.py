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
json_file_path = os.path.join(output_folder, "nonsensitive_fake_letters.json")

# List of non-sensitive topics
topics = [
    "a reminder about the upcoming community event at the town square.",
    "an update on your subscription to our monthly magazine.",
    "a notification that your local library has new books available.",
    "a special offer for a discount on your next purchase at our store.",
    "a confirmation that your favorite movie is now available on streaming platforms.",
    "a note about the weather forecast for the weekend.",
    "an invitation to join a free online course on photography.",
    "a suggestion to try a new recipe from our cooking blog.",
    "an announcement about a new park opening in your area.",
    "a friendly reminder to participate in our customer satisfaction survey.",
    "a reminder about the upcoming community cleanup event.",
    "an invitation to the annual book fair at the local library.",
    "an update on the latest features added to our mobile app.",
    "a schedule of free yoga classes at the community center.",
    "a list of top-rated hiking trails in your area.",
    "a preview of the new art exhibition opening this weekend.",
    "a special promotion for eco-friendly household products.",
    "a thank-you note for your continued subscription to our newsletter.",
    "a guide to organizing your home office for better productivity.",
    "a summary of popular travel destinations for the upcoming holiday season.",
    "a tip on improving your gardening skills.",
    "a short guide to basic photography techniques.",
    "an overview of beginner-friendly coding tutorials.",
    "a recipe for a quick and healthy breakfast.",
    "a list of must-read classic novels for book lovers.",
    "a beginner's guide to playing the guitar.",
    "a brief history of a famous monument in your city.",
    "a tutorial on creating DIY crafts from recycled materials.",
    "a brief explanation of how solar panels work.",
    "an introduction to basic meditation techniques.",
    "a list of family-friendly movies to watch this weekend.",
    "a fun quiz about your favorite TV show.",
    "a recommendation for the latest bestseller in mystery novels.",
    "a review of the top-rated board games this year.",
    "a guide to planning a budget-friendly weekend getaway.",
    "an announcement of a new season of a popular podcast.",
    "a list of the top-rated restaurants in your neighborhood.",
    "a guide to exploring hidden gems in your city.",
    "a list of classic video games that are still fun to play.",
    "an overview of upcoming festivals and fairs in your area.",
    "a guide to saving energy at home.",
    "a checklist for maintaining your car before a road trip.",
    "a reminder to check your smoke detector batteries.",
    "Tips for improving your email organization.",
    "a guide to preparing your garden for the winter season.",
    "a summary of the latest technological trends.",
    "a beginner’s guide to composting at home.",
    "an explanation of how to set up a budget-friendly home gym.",
    "a list of essential items for an emergency kit.",
    "a reminder to check for local road closures before traveling."
]


# Function to generate a fake letter
def generate_fake_letter(number_of_letters):
    # Generate data
    data = []

    for _ in range(number_of_letters):
        # Generate fake name and address
        first_name = fake.first_name()
        last_name = fake.last_name()
        phone_number = fake.basic_phone_number()
        company_name = fake.company()

        # Pick a random non-sensitive topic
        topic = random.choice(topics)

        # Construct the letter
        letter = f"""
        Dear {first_name} {last_name},

        We hope this letter finds you well. 
        We are writing to inform you about {topic}

        For any questions or further assistance, 
        feel free to contact us at:

        Mail: {fake.company_email()} 
        Business phone number: {phone_number}.


        Best regards,
        {fake.name()}
        {company_name}
        """

        data.append(letter)

    return data


# Save data to JSON file
with open(json_file_path, "w") as json_file:
    json.dump(generate_fake_letter(1), json_file, indent=4)

print(f"fake letters data saved to {json_file_path}")

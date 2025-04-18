import json
import random
from datetime import datetime, timedelta
import os

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "non_sensitive_events.json")

# List of sample event names and speakers
event_names = [
    "Tech Conference", "Music Festival", "Art Exhibition",
    "Science Fair", "Business Summit", "Gaming Expo"
]
speakers = ["Dr. Jane Smith", "Prof. John Doe", "Alex Lee", "Taylor Brown", "Jordan Kim", "Chris Patel"]


# Generate a random date within the next two years
def generate_random_date():
    start_date = datetime.now()
    end_date = start_date + timedelta(days=730)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime("%Y-%m-%d")


# Generate a random element
def generate_random_element():
    return {
        "Event Name": f"{random.choice(event_names)} {random.randint(2025, 2030)}",
        "Date": generate_random_date(),
        "Location": random.choice(["New York", "Los Angeles", "London", "Berlin", "Tokyo", "Sydney"]),
        "Number of Attendees": random.randint(50, 10000),
        "Keynote Speaker": random.choice(speakers)
    }


# Generate elements
data = [generate_random_element() for _ in range(50000)]

# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with non-sensitive elements saved as {file_path}.")

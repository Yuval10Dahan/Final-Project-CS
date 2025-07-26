import json
import random
import os

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "non_sensitive_geographic.json")

# List of sample cities and landmarks
cities = [
    {"City": "San Francisco", "Country": "USA", "Landmark": "Golden Gate Bridge"},
    {"City": "Paris", "Country": "France", "Landmark": "Eiffel Tower"},
    {"City": "Tokyo", "Country": "Japan", "Landmark": "Tokyo Tower"},
    {"City": "Sydney", "Country": "Australia", "Landmark": "Sydney Opera House"},
    {"City": "Rome", "Country": "Italy", "Landmark": "Colosseum"},
    {"City": "Cairo", "Country": "Egypt", "Landmark": "Pyramids of Giza"},
    {"City": "New York", "Country": "USA", "Landmark": "Statue of Liberty"}
]


def generate_random_element():
    location = random.choice(cities)
    return {
        "City": location["City"],
        "Country": location["Country"],
        "Landmark": location["Landmark"],
        "Latitude": round(random.uniform(-90, 90), 4),
        "Longitude": round(random.uniform(-180, 180), 4),
        "Temperature (°C)": round(random.uniform(-10, 40), 1),
        "Humidity (%)": random.randint(10, 100)
    }


# Generate 100,000 elements
data = [generate_random_element() for _ in range(50000)]

# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with non-sensitive elements saved as {file_path}.")

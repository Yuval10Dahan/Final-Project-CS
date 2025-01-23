import json
import random
import os

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "non_sensitive_info.json")

# Generate a list of random product names
product_names = ["Ergonomic Chair", "Wireless Mouse", "Mechanical Keyboard", "Standing Desk",
                 "Noise-Canceling Headphones", "Portable Monitor", "Webcam", "Gaming Mousepad"]

# Generate a list of random categories
categories = ["Furniture", "Electronics", "Office Supplies", "Gaming Accessories"]


def generate_random_element():
    return {
        "Name": f"Store-{random.randint(1000, 9999)}",
        "Product ID": f"PR-{random.randint(10000, 99999)}",
        "Product Name": random.choice(product_names),
        "Category": random.choice(categories),
        "Price": round(random.uniform(10.00, 500.00), 2),
        "Rating": round(random.uniform(1.0, 5.0), 1),
        "Stock Quantity": random.randint(1, 500)
    }


# Generate elements
data = [generate_random_element() for _ in range(1)]

# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with non-sensitive elements saved as non_sensitive_info.json.")

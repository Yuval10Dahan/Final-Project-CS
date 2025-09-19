import json
import random
from datetime import datetime, timedelta
import os
from faker import Faker

fake = Faker()

# Directory where the JSON file will be saved
output_folder = "json_files"

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Filepath for the JSON file
file_path = os.path.join(output_folder, "non_sensitive_reviews.json")

# Sample products and review texts
products = [
    "Wireless Earbuds", "Smartphone", "Gaming Laptop", "4K Monitor",
    "Mechanical Keyboard", "Bluetooth Speaker", "Fitness Tracker"
]
review_texts = [
    "The sound quality is excellent, and the battery lasts long enough for daily use.",
    "Great product for the price, but the build quality could be better.",
    "Exceeded my expectations! Highly recommended for anyone looking for performance.",
    "Good value for money, but the customer support was not very helpful.",
    "Amazing design and features, but the software needs improvement.",
    "The product arrived quickly and was well-packaged, but it didn't meet my expectations in terms of quality.",
    "Absolutely love it! The performance is top-notch, and it’s very easy to use.",
    "The instructions were unclear, and it took a while to figure out how to get it working properly.",
    "The battery life is incredible, lasting for days on a single charge.",
    "The screen resolution is stunning, but the colors feel slightly oversaturated.",
    "Its lightweight and portable, making it perfect for travel.",
    "Feels a bit overpriced for the features it offers, but it does the job.",
    "The app integration works seamlessly, and setup was a breeze.",
    "Build quality is solid, but the design feels a bit outdated.",
    "The customer service was excellent and resolved my issue quickly.",
    "The product overheats after extended use, which is disappointing.",
    "The sound is crystal clear, and the bass is surprisingly powerful for its size.",
    "I appreciate the eco-friendly packaging, but the product itself could be more durable.",
    "Its a good starter option, but advanced users might find it lacking in features.",
    "The wireless connectivity is flawless, with no drops or interruptions during use."
]


def generate_random_review():
    return {
        "Product Name": random.choice(products),
        "Review ID": f"RV-{random.randint(10000, 99999)}",
        "Reviewer Name": fake.name(),
        "Review Date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
        "Rating": round(random.uniform(1.0, 5.0), 1),
        "Review Text": random.choice(review_texts),
        "Helpful Votes": random.randint(0, 500)
    }


# Generate reviews
data = [generate_random_review() for _ in range(50000)]

# Write to JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"JSON file with non-sensitive reviews saved as {file_path}.")

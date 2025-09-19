from PIL import Image, ImageDraw, ImageFont
import os
import json

# Directory to save review images
image_output_folder = "review_images"
os.makedirs(image_output_folder, exist_ok=True)

# Load reviews from the JSON file
file_path = os.path.join("json_files", "non_sensitive_reviews.json")
with open(file_path, "r") as file:
    reviews = json.load(file)

# Font settings (adjust the path to your system)
font_path = "arial.ttf"  # Change this to the path of your desired font
font_size = 16

try:
    font = ImageFont.truetype(font_path, font_size)
except:
    print("Font not found. Using default font.")
    font = ImageFont.load_default()


# Function to create an image of a review
def create_review_image(review, index):
    img_width, img_height = 900, 400
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)

    # Create a blank image
    img = Image.new("RGB", (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    # Prepare review text
    review_text = (
        f"Reviewer: {review['Reviewer Name']}\n"
        f"ID: {review['Review ID']}\n"
        f"Date: {review['Review Date']}\n"
        f"Rating: {review['Rating']} \n\n"
        f"Product: {review['Product Name']}\n\n"
        f"Review: \n{review['Review Text']}\n\n\n\n"
        f"Helpful Votes: {review['Helpful Votes']}"
    )

    # Draw the text
    margin = 20
    current_height = margin
    for line in review_text.split("\n"):
        draw.text((margin, current_height), line, fill=text_color, font=font)
        current_height += 25

    # Save the image
    img.save(os.path.join(image_output_folder, f"review_{index}.png"))
    print(f"review_{index}.png saved")


# Generate images for the first 100 reviews (adjust as needed)
for i, review in enumerate(reviews[:]):
    create_review_image(review, i)

print(f"Review images saved in folder: {image_output_folder}")

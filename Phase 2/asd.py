from faker import Faker
import os
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import os
 

fake = Faker()
# Path to a valid TTF font you have
TITLE_FONT_PATH = "arial.ttf"
BODY_FONT_PATH = "arial.ttf"

def generate_sensitive_text():
    # Example placeholders for sensitive info: SSN, credit card, etc.
    id = fake.ssn()  # "123-45-6789"
    cc_number = fake.credit_card_number()  # e.g., "4111 1111 1111 1111"

    return (
        f"Dear {fake.name()},\n\n"
        f"We are writing to confirm your identity. "
        f"Your ID on file is {id}, and your credit card on file is {cc_number}.\n"
        "Please keep this information safe.\n"
        "\nSincerely,\nSensitive Info Dept.\n"
    )

def generate_not_sensitive_text():
    # Generic letter, no personal info
    return (
        f"Dear {fake.name()},\n\n"
        "We hope this letter finds you well. "
        "Attached is the agenda for next week's team meeting. "
        "No private information is shared in this correspondence.\n"
        "\nBest regards,\nYour Company\n"
    )


def generate_formal_letter_image(
    output_path,
    sender_name="XYZ Corp",
    sender_subtext="Corporate Communications",
    body_text="Hello,\nThis is sample text.",
    signature_name="John Doe, Manager",
    width=800,
    height=1000
):
    """
    Generates a 'formal letter' style image containing the given body_text
    and saves it to output_path as a PNG.
    """

    # 1. Create a blank white image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # 2. Load fonts (adjust paths & sizes)
    title_font = ImageFont.truetype(TITLE_FONT_PATH, 32)
    subtitle_font = ImageFont.truetype(BODY_FONT_PATH, 20)
    body_font = ImageFont.truetype(BODY_FONT_PATH, 24)

    x_margin = 50
    y_offset = 50

    # 3. Sender name (title)
    draw.text((x_margin, y_offset), sender_name, fill="black", font=title_font)
    _, top, _, bottom = draw.textbbox((0, 0), sender_name, font=title_font)
    y_offset += (bottom - top) + 10

    # 4. Sender subtext
    draw.text((x_margin, y_offset), sender_subtext, fill="black", font=subtitle_font)
    _, top, _, bottom = draw.textbbox((0, 0), sender_subtext, font=subtitle_font)
    y_offset += (bottom - top) + 30

    # 5. Add date
    current_date = datetime.now().strftime("%B %d, %Y")
    draw.text((x_margin, y_offset), current_date, fill="black", font=subtitle_font)
    _, top, _, bottom = draw.textbbox((0, 0), current_date, font=subtitle_font)
    y_offset += (bottom - top) + 30

    # 6. Body text (wrap lines)
    wrapped_body = textwrap.fill(body_text, width=60)

    for line in wrapped_body.split("\n"):
        draw.text((x_margin, y_offset), line, fill="black", font=body_font)
        _, top, _, bottom = draw.textbbox((0, 0), line, font=body_font)
        y_offset += (bottom - top) + 5

    y_offset += 50

    # 7. Closing / Signature
    closing_text = f"Sincerely,\n{signature_name}"
    for line in closing_text.split("\n"):
        draw.text((x_margin, y_offset), line, fill="black", font=body_font)
        _, top, _, bottom = draw.textbbox((0, 0), line, font=body_font)
        y_offset += (bottom - top) + 5

    # 8. Save the image
    img.save(output_path, format="PNG")
    print(f"Saved letter to {output_path}")


def create_sensitive_dataset(num_images=5, fraction_sensitive=0.4):
    """
    Generates a dataset of 'formal letter' images in two categories:
    - 'sensitive': contains personal info (SSN, credit card, etc.)
    - 'not_sensitive': normal text without private data.
    """

    # 1. Make label subfolders
    os.makedirs("dataset/sensitive", exist_ok=True)
    os.makedirs("dataset/not_sensitive", exist_ok=True)

    num_sensitive = int(num_images * fraction_sensitive)
    num_not_sensitive = num_images - num_sensitive

    # 2. Generate 'sensitive' images
    for i in range(num_sensitive):
        filename = f"letter_sensitive_{i+1:05d}.png"
        output_path = os.path.join("dataset", "sensitive", filename)

        # Use your function that returns text with sensitive fields
        body_text = generate_sensitive_text()

        # Generate and save the image
        generate_formal_letter_image(output_path, body_text=body_text)

    # 3. Generate 'not_sensitive' images
    for i in range(num_not_sensitive):
        filename = f"letter_not_sensitive_{i+1:05d}.png"
        output_path = os.path.join("dataset", "not_sensitive", filename)

        # Use your function that returns normal text
        body_text = generate_not_sensitive_text()

        # Generate and save the image
        generate_formal_letter_image(output_path, body_text=body_text)

    print(f"\nCreated {num_sensitive} sensitive + {num_not_sensitive} not_sensitive images.")

# Example usage
if __name__ == "__main__":
    create_sensitive_dataset(num_images=10, fraction_sensitive=0.5)

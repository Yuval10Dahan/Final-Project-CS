from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_sensitive_text_image(
    name="John Doe",
    ssn="123-45-6789",
    card_number="4111 1111 1111 1111",
    address="123 Main St, Springfield, USA",
    font_path="arial.ttf",
    font_size=24,
    output_file="sensitive_text_image.png"
):
    # 1. Create the text content
    text_content = f"""
    SENSITIVE INFO:

    Name: {name}
    ID: {ssn}
    Credit Card: {card_number}
    Address: {address}
    """

    # 2. Wrap the text so it doesn’t run off the edges
    wrapped_text = textwrap.fill(text_content, width=50)

    # 3. Load font
    font = ImageFont.truetype(font_path, font_size)

    # 4. Create a temporary image to measure the text
    temp_img = Image.new("RGB", (1, 1))
    draw_temp = ImageDraw.Draw(temp_img)

    # 5. Use 'textbbox' to get the bounding box of the text
    #    textbbox returns (left, top, right, bottom)
    text_left, text_top, text_right, text_bottom = draw_temp.textbbox(
        (0, 0),
        wrapped_text,
        font=font
    )
    text_width = text_right - text_left
    text_height = text_bottom - text_top

    # 6. Create a new image with some padding
    padding = 20
    img_width = text_width + (2 * padding)
    img_height = text_height + (2 * padding)

    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    # 7. Draw the text on the final image
    draw.text(
        (padding, padding),
        wrapped_text,
        font=font,
        fill="black"
    )

    # 8. Save the image
    img.save(output_file)
    print(f"Image saved to {output_file}")


generate_sensitive_text_image()
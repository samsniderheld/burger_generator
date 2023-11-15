"""
Basic Utility Functions

This module offers utility functions for various processing tasks. Key functionalities include:
- Blending images with masks.
- Reading image-related details from text files.
- Loading images which is done frequently.
- Drawing text on images
"""
from PIL import Image, ImageFilter,ImageDraw, ImageFont


def blend_image(inpainted, original, mask, blur=3):
    """Blend two images together using a mask and an optional blur."""
    mask = mask.convert("L")
    # Apply blur
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    # Blend images together
    return Image.composite(inpainted.convert('RGBA'), original.convert('RGBA'), mask).convert('RGBA')

def read_ingredients_from_txt(file_path):
    """Read ingredients from a text file and return them as a list."""
    ingredients = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            ingredients.append(line.strip())  # strip to remove any trailing whitespace and newline characters
    return ingredients

def load_img_for_sdxl(path):
    """loads the images for the inpainting pipeline"""
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((1024,1024))
    return img

def draw_text(draw, text, position, font, max_width):
    """Draw the text on the image with word wrapping."""
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and font.getsize(line + words[0])[0] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line)

    # Draw each line of text
    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=(0,0,0))
        y += font.getsize(line)[1]

def add_text_to_image(image, text, font_path='Lobster-Regular.ttf', font_size=50, max_width=400):
    """Wrapper function for the draw_text_function"""
    # Load the image
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Starting position for the text
    position = (50, 50)

    # Draw the text
    draw_text(draw, text, position, font, max_width)



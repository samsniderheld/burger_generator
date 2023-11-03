"""
Image Processing Utility Functions

This module offers utility functions for various image processing tasks. Key functionalities include:
- Blending images with masks.
- Reading image-related details from text files.

Functions:
    blend_image(inpainted, original, mask, blur=3):
        Blends an inpainted image with an original using a mask.

    read_ingredients_from_txt(file_path):
        Extracts ingredient names from a text file.
"""
from PIL import Image, ImageFilter

# Blend two images together using a mask and an optional blur.
def blend_image(inpainted, original, mask, blur=3):
    mask = Image.fromarray(mask)
    mask = mask.convert("L")
    # Apply blur
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    # Blend images together
    return Image.composite(inpainted.convert('RGBA'), original.convert('RGBA'), mask).convert('RGBA')

# Read ingredient image paths from a text file and return them as a list.
def read_ingredients_from_txt(file_path):
    """Read ingredients from a text file and return them as a list."""
    ingredients = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            ingredients.append(line.strip())  # strip to remove any trailing whitespace and newline characters
    return ingredients

def load_img_for_sdxl(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((1024,1024))
    return img
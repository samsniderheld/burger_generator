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
import random
from PIL import Image, ImageFilter

# Blend two images together using a mask and an optional blur.
def blend_image(inpainted, original, mask, blur=3):
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

#loads the images for the inpainting pipeline
def load_img_for_sdxl(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((1024,1024))
    return img

#enforces a 2:1 BK ingredients to random ingredients ratio
def enforce_standard_ingredient_ratio(all_ingredients, standard_ingredients, ingredients_num):
    total_count = ingredients_num

    if total_count  == 1:
        standard_count = 1
        random_count = 0
    elif total_count ==2:
        standard_count = 1
        random_count = 1
    else:
        standard_count = 2 * total_count // 3
        random_count = total_count - standard_count
    
    selected_standard = random.sample(standard_ingredients, standard_count)
    selected_random = random.sample(all_ingredients, random_count)
    
    ingredients = selected_standard + selected_random

    return ingredients

def contstruct_prompt_from_ingredient_list(ingredients):
        ingredient_string = "".join([f"{ingredient}++, " for ingredient in ingredients])
        prompt = f'A whopper with a beef patty and {args.num_ingredients} extra ingredients. {ingredient_string[:-1]}.'
        return prompt

#if we are using standard ingredients, make sure that we are negatively prompting correctly
def construct_negative_prompt_for_standard_ingredients(ingredients):

    new_basic_ingredients = []

    for ing in standard_ingredients:
            if(ing not in ingredients):
                new_basic_ingredients.append(ing)

    negative_prompt = ["poor quality, unappetizing, " + "".join([f"{ing}, " for ing in new_basic_ingredients])]

    return negative_prompt
"""
Image Processing Utility Functions

This module contains a set of utility functions designed for image processing and manipulation.
It provides functionalities to generate noise profiles, create layered images, blend, and overlay images, 
and read image ingredients from CSV files.

Functions:
    - `generate_noise_profile`: Generates a 1D array resembling a mountain profile based on width.
    - `multi_layer_image`: Produces an image with multiple layers separated by mountain profiles.
    - `overlay_images`: Overlays one image over another.
    - `composite_ingredients`: Composites multiple ingredients images into a burger template.
    - `blend_image`: Blends an inpainted image with an original using a mask.
    - `read_ingredients_from_csv`: Reads ingredient names from a CSV file.
"""
import csv
import cv2
import numpy as np
import noise
from PIL import Image, ImageFilter

def generate_noise_profile(width, base_y, layer_index, amplitude=10, scale=100.0,octaves=2):
    """Generate a 1D array resembling a mountain profile based on the width."""
    # Seed the noise with layer index to make each boundary different
    mountain = [base_y + int(amplitude * noise.pnoise1(x / scale + layer_index * 10, octaves=octaves)) for x in range(width)]
    return mountain

def multi_layer_image(width, height, num_layers,amplitude=10, scale=100.0, octaves=1):
    """Generate an image with multiple layers separated by mountain profiles."""
    img = np.zeros((height, width, 3), dtype=np.uint8)  # 3 for RGB
    
    # Define tones for each layer (for demonstration, using grayscale tones)
    tones = np.linspace(0, 255, num_layers, dtype=np.uint8)
    
    # Calculate the height of each layer
    layer_height = height // num_layers
    
    # Assign the tones
    for x in range(width):
        for layer in range(num_layers):
            # Calculate base y for this layer
            base_y = layer * layer_height
            
            if layer != num_layers - 1:
                mountain_y = generate_noise_profile(width, base_y + layer_height, layer, amplitude, scale, octaves)[x]
                for y in range(layer_height):
                    if y + base_y < mountain_y:
                        img[y + base_y, x] = tones[layer]
                    else:
                        img[y + base_y, x] = tones[layer + 1]
            else:
                img[base_y:base_y+layer_height, x] = tones[layer]
    
    return img,tones

def overlay_images(background, overlay):
    # Resize overlay image to fit the background
    overlay = overlay.resize(background.size, Image.ANTIALIAS)

    # Composite the images
    combined = Image.alpha_composite(background.convert("RGBA"), overlay)

    # Convert to 'RGB' before saving as JPEG
    combined = combined.convert("RGB")

    return combined

def composite_ingredients(ingredient_1, mask_1, ingredient_2,mask_2,burger_template):
    
    # Resize images to 512x512
    dim = (512, 512)
    base_array = cv2.resize(np.uint8(ingredient_1), dim)
    mask_array = cv2.resize(np.uint8(mask_1), dim)
    second_array = cv2.resize(np.uint8(ingredient_2), dim)
    second_mask_array = cv2.resize(np.uint8(mask_2), dim)
    target_array = cv2.resize(np.uint8(burger_template), dim)

    # Step 4: Identify white areas in the mask and copy corresponding pixels from the base to the target image
    white_area = (mask_array[:,:,0] == 255) & (mask_array[:,:,1] == 255) & (mask_array[:,:,2] == 255)
    target_array[white_area] = base_array[white_area]

    white_area_2 = (second_mask_array[:,:,0] == 255) & (second_mask_array[:,:,1] == 255) & (second_mask_array[:,:,2] == 255)
    target_array[white_area_2] = second_array[white_area_2]

    target_array = Image.fromarray(target_array)

    return target_array

def blend_image(inpainted, original, mask, blur=3):
    mask = mask.convert("L")
    # Apply blur
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    # Blend images together
    return Image.composite(inpainted.convert('RGBA'), original.convert('RGBA'), mask).convert('RGBA')


def read_ingredients_from_csv(file_path):
    """Read ingredients from a CSV file and return them as a list."""
    ingredients = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header if there is any
        for row in reader:
            ingredients.append(row[0])
    return ingredients
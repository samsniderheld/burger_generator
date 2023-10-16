"""
Image Processing Utility Functions

This module offers utility functions for various image processing tasks. Key functionalities include:
- Generating noise profiles which resemble mountain ranges.
- Constructing images with multiple layers.
- Compositing multiple images based on specific conditions.
- Blending images with masks.
- Reading image-related details from text files.

Functions:
    generate_noise_profile(width, base_y, layer_index, amplitude=10, scale=100.0, octaves=2):
        Produces a 1D array resembling a mountain profile.

    multi_layer_img(width, height, gen_space_x, layer_height, num_layers):
        Generates an image with multiple layers separated by mountain profiles.

    generate_noisy_rectangle_path(center, width, height, amplitude=20, scale=0.2):
        Constructs a noisy path for a rectangle.

    apply_noisy_mask(img, gen_space_x, gen_space_y):
        Applies a noisy mask to an image based on specified dimensions.

    generate_template_and_mask(layers, overlay_top, overlay_bottom):
        Generates a layered template and a corresponding mask.

    composite_ingredients(ingredients, template, template_values, dims=512):
        Composites multiple ingredients images into a layered template.

    blend_image(inpainted, original, mask, blur=3):
        Blends an inpainted image with an original using a mask.

    read_ingredients_from_txt(file_path):
        Extracts ingredient names from a text file.
"""
import cv2
import numpy as np
import random
from noise import pnoise1
from PIL import Image, ImageFilter

layer_amplitude_min = 40
layer_amplitude_max = 70
layer_scale_min = 150
layer_scale_max = 180

side_amplitude_min = 10
side_amplitude_max = 25
side_scale_min = 70
side_scale_max = 80

layer_height = 60

# Generate a noise profile based on the given parameters.
# This profile simulates the profile of a mountain.
def generate_noise_profile(width, base_y, layer_index, amplitude=10, scale=100.0,octaves=2):
    """Generate a 1D array resembling a mountain profile based on the width."""
    mountain = [base_y + int(amplitude * pnoise1(x/ scale + layer_index*10, octaves=octaves)) for x in range(width)]
    return mountain

# Generate an image with multiple layers, where each layer simulates a mountain range.
def multi_layer_img(width, height, gen_space_x, layer_height, num_layers):
    gen_space_y = num_layers*layer_height
    img = np.zeros((height, width, 3), dtype=np.uint8)  # 3 for RGB
    tones = np.linspace(50, 200, num_layers, dtype=np.uint8)
    left_over_height = height - gen_space_y
    start_y = int(left_over_height / 2)

    for layer in range(num_layers):
      amplitude=random.randint(layer_amplitude_min,layer_amplitude_max)
      scale=random.randint(layer_scale_min, layer_scale_max)
      octaves=4
      base_y = (layer * layer_height) + start_y
      mountain = generate_noise_profile(width, base_y + layer_height, layer, amplitude, scale, octaves)
      for x in range(width):
            if layer != num_layers - 1:
                mountain_y = mountain[x]
                
                for y in range(layer_height):
                    if y + base_y < mountain_y:
                        img[y + base_y, x] = tones[layer]
                    else:
                        img[y + base_y, x] = tones[layer + 1]
            else:
                img[base_y:base_y + layer_height, x] = tones[layer]

    masked_img = apply_noisy_mask(img, gen_space_x, gen_space_y)

    return masked_img,tones,start_y

# Generate a noisy rectangle path. This path is used for masking.
def generate_noisy_rectangle_path(center, width, height, amplitude=20, scale=0.2):
    half_width = width // 2
    half_height = height // 2
    
    # Coordinates of the four corners of the rectangle
    top_left = (center[0] - half_width, center[1] - half_height)
    top_right = (center[0] + half_width, center[1] - half_height)
    bottom_left = (center[0] - half_width, center[1] + half_height)
    bottom_right = (center[0] + half_width, center[1] + half_height)
    
    # Generate noisy paths for each edge
    left_path = np.linspace(top_left, bottom_left, height)
    right_path = np.linspace(top_right, bottom_right, height)
    
    # Apply noise
    left_path[:, 0] -= amplitude * np.array([pnoise1(y / scale + 30) for y in np.linspace(0, height, height)])
    right_path[:, 0] += amplitude * np.array([pnoise1(y / scale + 30) for y in np.linspace(0, height, height)])
    
    # Concatenate the paths to form a continuous loop
    path = np.vstack(( right_path, left_path[::-1]))  # [::-1] to reverse order
    
    return path

# Apply a noisy mask to the given image.
def apply_noisy_mask(img, gen_space_x, gen_space_y):
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Black mask

    center = (width // 2, height // 2)
    amplitude = random.randint(side_amplitude_min,
                               side_amplitude_max)
    scale = random.randint(side_scale_min,side_scale_max)
    path = generate_noisy_rectangle_path(center, gen_space_x, gen_space_y,amplitude,scale)
    
    cv2.fillPoly(mask, [path.astype(np.int32)], 255)  # Fill with white (255) inside the noisy ellipse
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img, mask_3_channel)
    result_with_white_bg = result.copy()
    area_black = (result_with_white_bg[:,:,0] == 0) & (result_with_white_bg[:,:,1] == 0) & (result_with_white_bg[:,:,2] == 0)
    result_with_white_bg[area_black] = 255

    return result_with_white_bg

# Generate a template image and a mask based on provided overlays.
def generate_template_and_mask(layers,overlay_top,overlay_bottom):
    width = 512
    height = 512
    x_mod = .8
    gen_space_x = int(width * x_mod)
    top_overlay_mod = 120
    bottom_overlay_mod = 20
    
    #generates template
    img, values, start_y = multi_layer_img(width, height, gen_space_x, 
        layer_height, layers)
    
    #generates mask
    mask = img.copy()
    area_white = (mask[:,:,0] == 255) & (mask[:,:,1] == 255) & (mask[:,:,2] == 255)
    mask[area_white] = 0
    area_shade = (mask[:,:,0] != 0) & (mask[:,:,1] != 0) & (mask[:,:,2] != 0)
    mask[area_shade] = 255

    # Overlay the top image
    composite = Image.fromarray(img)
    composite.paste(overlay_top, (0, start_y-top_overlay_mod), overlay_top)
    composite.paste(overlay_bottom, 
                    (0, start_y+(layers*layer_height-bottom_overlay_mod)), overlay_bottom)

    return composite, values, mask

# Composite multiple images (ingredients) onto a template based on a mask.
def composite_ingredients(ingredients,template,template_values,dims=512):
    dim = (dims,dims)
    template = np.array(template)
    
    for i,ingredient in enumerate(ingredients):
        ingredient = np.array(ingredient)
    
        # Identify areas in the mask and copy corresponding pixels from the base to the target image
        area = (template[:,:,0] == template_values[i]) & (template[:,:,1] == template_values[i]) & (template[:,:,2] == template_values[i])
        template[area] = ingredient[area]

    template = Image.fromarray(template)

    return template

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

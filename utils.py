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
    # random_seed = random.uniform(-10, 10)
    random_seed = 0
    mountain = [base_y + int(amplitude * noise.pnoise1(x/ scale + layer_index *10, octaves=octaves)+random_seed) for x in range(width)]
    return mountain

def multi_layer_image(width, height, num_layers,amplitude=10, scale=100.0, octaves=1):
    """Generate an image with multiple layers separated by mountain profiles."""
    img = np.zeros((height, width, 3), dtype=np.uint8)  # 3 for RGB
    num_layers+=2
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
  
    return img,tones[1:-1]

def generate_noisy_circle_path(center, radius, width, amplitude=20, scale=0.2):
    angles = np.linspace(0, 2 * np.pi, width)
    x_coords = center[0] + radius * np.cos(angles) + amplitude * np.array([noise.pnoise1(angle / scale) for angle in angles])
    y_coords = center[1] + radius * np.sin(angles) + amplitude * np.array([noise.pnoise1(angle / scale + 10) for angle in angles])  # offset to create different noise
    path = np.vstack((x_coords, y_coords)).T
    return path

def apply_noisy_mask(image):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Black mask
    white_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White image

    center = (width // 2, height // 2)
    radius = min(width, height) // 2.3  # Adjust as necessary
    path = generate_noisy_circle_path(center, radius, width)

    cv2.fillPoly(mask, [path.astype(np.int32)], 255)  # Fill with white (255) inside the noisy circle

    # Convert the mask to 3 channel
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    # Masking the image
    result = cv2.bitwise_and(image, mask_3_channel).astype(np.uint8)

    # Use cv2.bitwise_or to combine the result with the white image
    result_with_white_bg = cv2.bitwise_not(result, white_img).astype(np.uint8)
    
    #this bitwise_not function introduces a bug which needs to be fixed.
    input_vals = np.unique(image)
    output_vals = np.unique(result_with_white_bg)
    if  not np.array_equal(input_vals,output_vals):
      for val in input_vals:
        area = (result_with_white_bg[:,:,0] == val+1) & (result_with_white_bg[:,:,1] == val+1) & (result_with_white_bg[:,:,2] == val+1)
        result_with_white_bg[area] = val
    
    return result_with_white_bg

def generate_template(layers,overlay):
  img, values = multi_layer_image(512, 512, layers+1, amplitude=100, scale=150, octaves=5)
  cv2.imwrite("layers.png",img)
  masked_image = apply_noisy_mask(img)
  masked_image = Image.fromarray(masked_image)

  separation = 60

  # Overlay the top image
  composite = masked_image.copy()
  composite.paste(overlay, (0, 0), overlay)

  return composite, values

def overlay_images(background, overlay):
    # Resize overlay image to fit the background
    overlay = overlay.resize(background.size, Image.ANTIALIAS)

    # Composite the images
    combined = Image.alpha_composite(background.convert("RGBA"), overlay)

    # Convert to 'RGB' before saving as JPEG
    combined = combined.convert("RGB")

    return combined

def composite_ingredients(ingredients,template,template_values,dims=512):
    combined = zip(ingredients,template_values)
    dim = (dims,dims)
    template = np.array(template)
    
    for i,ingredient in enumerate(ingredients):
        ingredient = np.array(ingredient)
    
        # Identify areas in the mask and copy corresponding pixels from the base to the target image
        area = (template[:,:,0] == template_values[i]) & (template[:,:,1] == template_values[i]) & (template[:,:,2] == template_values[i])
        template[area] = ingredient[area]

    template = Image.fromarray(template)

    return template

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
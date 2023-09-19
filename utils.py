import csv
from PIL import Image, ImageFilter

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
    base_array = cv2.resize(ingredient_1, dim)
    mask_array = cv2.resize(mask_1, dim)
    second_array = cv2.resize(ingredient_2, dim)
    second_mask_array = cv2.resize(mask_2, dim)
    target_array = cv2.resize(burger_template, dim)

    # Step 4: Identify white areas in the mask and copy corresponding pixels from the base to the target image
    white_area = (mask_array[:,:,0] == 255) & (mask_array[:,:,1] == 255) & (mask_array[:,:,2] == 255)
    target_array[white_area] = base_array[white_area]

    white_area_2 = (second_mask_array[:,:,0] == 255) & (second_mask_array[:,:,1] == 255) & (second_mask_array[:,:,2] == 255)
    target_array[white_area_2] = second_array[white_area_2]

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
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
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
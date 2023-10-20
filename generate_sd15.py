
import os
import random
import time
import cv2
import numpy as np
from PIL import Image
from arg_parser import parse_args
from pipelines.pipelines import (ControlNetPipeline, Img2ImgPipeline,
                                Img2ImgSDXLPipeline)
from utils import (blend_image, composite_ingredients, 
                  generate_template_and_mask, read_ingredients_from_txt)

# Parse arguments from command line or script input
args = parse_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load base images
control_net_img = Image.open(args.input_texture)
overlay_top = Image.open(os.path.join(args.overlay_dir, "top.png")).convert("RGBA")
overlay_bottom = Image.open(os.path.join(args.overlay_dir, "bottom.png")).convert("RGBA")

# Initialize pipelines
img2img_pipe = Img2ImgPipeline(args.base_img2img_model)


for i in range(args.num_samples):
    start_time = time.time()

    all_ingredients = read_ingredients_from_txt(args.txt_file)
    num_ingredients = random.randint(1,3)
    ingredients = random.sample(all_ingredients,num_ingredients)
   
    # Generate burger template
    template, template_values, mask = generate_template_and_mask(
        len(ingredients), overlay_top, overlay_bottom
    )
    # Generate burger image with ingredients
    burger_ingredient_string = "".join([f"{ingredient}++, " for ingredient in ingredients])
    burger_prompt = (
        f"image a burger with a {burger_ingredient_string} photorealistic photography, 8k uhd, "
        "full framed, photorealistic photography, dslr, soft lighting, high quality, Fujifilm XT3\n\n"
    )

    print(burger_prompt)


    directory_path = args.texture_dir
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    file_urls = sorted(
        [
            os.path.join(directory_path, f) for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f)) and 
            os.path.splitext(f)[1].lower() in image_extensions
        ],
        key=str.casefold
    )
    all_ingredients = [os.path.basename(file).split("_")[0] for file in file_urls]
    all_ingredients = list(set(all_ingredients))
    ingredient_textures = [Image.open(random.choice([path for path in file_urls if ingredient in path])) for ingredient in ingredients]

    # Composite and blend images to get the final output
    input_img = composite_ingredients(ingredient_textures[::-1], template, template_values)
    img = img2img_pipe.generate_img(burger_prompt, input_img, args.img2img_strength, args.burger_steps, args.cfg_scale)
    img = blend_image(img, input_img, mask, args.mask_blur)

    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)




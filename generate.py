"""
generate.py

This script facilitates the generation of composite images, particularly burgers, 
based on a set of given or randomly chosen ingredients. It is designed to interface 
with specific pipelines (ControlNetPipeline and Img2ImgPipeline) for image 
generation. The process is conducted in the following order:

1. Parse input arguments which determine the behavior of the script.
2. Initialize necessary pipelines for image generation.
3. Load base images that will be used throughout the process.
4. In a loop for the desired number of samples:
    a. Determine the ingredients either from a given list or by random selection 
       from a provided text file.
    b. If texture generation is flagged, generate texture images for each ingredient.
    c. If burger image generation is flagged:
        i. Use a combination of generated or pre-existing textures to produce a composite image.
        ii. Generate the final burger image.
        iii. Save the burger image and a composite showcasing the image generation pipeline.

Key Features:
    - Allows for dynamic selection of ingredients from a text file.
    - Can generate texture images based on ingredient names.
    - Constructs composite burger images from ingredient textures.
    - Integrates with custom image generation pipelines.


Usage:
    !python generate.py \
    --input_texture "input_templates/01.jpg" \
    --base_texture_model "saved_model_offset_noise_realitic_vision" \
    --base_img2img_model "saved_model_realitic_vision_offset_noise_imgs2" \
    --output_dir "burger_outputs/" \
    --texture_steps 20 \
    --burger_steps 100 \
    --img2img_strength .5 \
    --controlnet_str .7 \
    --mask_blur 3 \
    --ingredients "bright salmon" "dark chocolate" "raspberries" "avocado"\
    --num_samples 1 \
    --gen_texture \
    --gen_burger
"""


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

# Initialize pipelines based on the flags
if args.gen_texture:
    controlnet_pipe = ControlNetPipeline(args.base_texture_model, args.controlnet_path)
if args.gen_burger:
    if args.use_SDXL:
        img2img_pipe = Img2ImgSDXLPipeline(args.base_img2img_model)
    else:
        img2img_pipe = Img2ImgPipeline(args.base_img2img_model)

# Main loop to generate images
for i in range(args.num_samples):
    start_time = time.time()

    # Load ingredients from a text file or directly from arguments
    if args.txt_file:
        ingredients = read_ingredients_from_txt(args.txt_file)
    else:
        ingredients = args.ingredients

    # Generate texture images for each ingredient if the flag is set
    if args.gen_texture:
        textures = []
        for ingredient in ingredients:
            # Generate texture image
            prompt = (
                f"food-texture, a 2D texture of {ingredient}+++, layered++, side view, "
                "full framed, photorealistic photography, 8k uhd, dslr, soft lighting, "
                "high quality, Fujifilm XT3\n\n"
            )
            texture = controlnet_pipe.generate_img(
                prompt, control_net_img, args.controlnet_str, args.dims, 
                args.dims, args.texture_steps, args.cfg_scale
            )
            # Append texture for burger image or save texture image directly
            if args.gen_burger: 
                textures.append(texture)
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"The script took {elapsed_time:.2f} seconds to execute.")
                out_img = cv2.cvtColor(np.uint8(texture), cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{args.output_dir}/{ingredient}_{i:04d}.jpg", out_img)

    # Generate final burger image if the flag is set
    if args.gen_burger:
        # Load ingredients from a text file or directly from arguments
        if args.txt_file:
            all_ingredients = read_ingredients_from_txt(args.txt_file)
            num_ingredients = random.randint(1,4)
            ingredients = random.choices(all_ingredients,k=num_ingredients)
        else:
            ingredients = args.ingredients
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

        # Decide on the source of the ingredient textures: generated or from directory
        if args.gen_texture:
            ingredient_textures = textures[::-1]
        else:
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

        #save a composite image showing the pipeline
        pipeline_img = np.hstack([template, input_img, img.convert('RGB')])
        pipeline_img = cv2.cvtColor(np.uint8(pipeline_img), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"pipeline_img.jpg", pipeline_img)



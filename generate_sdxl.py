import os
import random
import time
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

from arg_parser import parse_sdxl_args
from pipelines.pipelines import (ControlnetSDXLPipeline,InpaintingSDXLPipeline)
from utils import load_img_for_sdxl, read_ingredients_from_txt

font = ImageFont.truetype("OpenSans-Regular.ttf", 30)
# Parse arguments from command line or script input
args = parse_sdxl_args()

if args.food_list != None:

    random_ingredients = read_ingredients_from_txt("food_list.txt")

if args.use_standard_ingredients:
    standard_ingredients = ["lettuce", "tomatoes", "pickles", "onions", 
                            "ketchup", "cheese", "bacon", "extra patty", 
                            "mayonaise"]

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Initialize pipelines based on the flags
if args.pipeline_type == 'inpainting':
    sdxl_pipe  = InpaintingSDXLPipeline(args.sdxl_model)
elif args.pipeline_type == 'controlnet':
    sdxl_pipe  = ControlnetSDXLPipeline(args.sdxl_model)

for i in range(args.num_samples):

    start_time = time.time()

    if args.pipeline_type == 'inpainting':

        if args.food_list != None:

            if args.use_standard_ingredients:

                total_count = args.num_ingredients

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
                selected_random = random.sample(random_ingredients, random_count)
                
                ingredients = selected_standard + selected_random

                # standard_string = "".join([f"{ingredient}, " for ingredient in selected_standard])
                # random_string = "".join([f"{ingredient}, " for ingredient in selected_random])
                # prompt = f'A whopper with a beef patty and {args.num_ingredients} extra ingredients. ({standard_string[:-1]})++ , ({random_string[:-1]})++ .'
                ingredient_string = "".join([f"{ingredient}++, " for ingredient in ingredients])
                prompt = f'A whopper with a beef patty and {args.num_ingredients} extra ingredients. {ingredient_string[:-1]}.'
                
                new_basic_ingredients = []

                for ing in standard_ingredients:
                    if(ing not in ingredients):
                        new_basic_ingredients.append(ing)

                negative_prompt = ["poor quality, unappetizing, " + "".join([f"{ing}, " for ing in new_basic_ingredients])]
            else:

                ingredients = random.sample(random_ingredients, args.num_ingredients)
                ingredient_string = "".join([f"{ingredient}++, " for i, ingredient in enumerate(ingredients,1)])
                prompt = f'A whopper with a beef patty and {args.num_ingredients} extra ingredients. {ingredient_string[:-1]}.'
                negative_prompt = args.negative_prompt
        else:

            prompt = args.prompt
            negative_prompt = args.negative_prompt

        mask_num = 5

        path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
        base_img = load_img_for_sdxl(path)

        mask_path = os.path.join(args.template_dir,f"{mask_num}_ingredient_mask.png")
        mask_img = load_img_for_sdxl(mask_path)   

        img = sdxl_pipe.generate_img(
            prompt, 
            negative_prompt,
            base_img,
            mask_img,
            1,
            args.cfg_scale,
            args.steps,
            True
        )

    elif args.pipeline_type == 'controlnet':

        mask_num = random.randint(1,5)

        path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
        base_img = load_img_for_sdxl(path)

        img = sdxl_pipe.generate_img(
            args.prompt,
            args.negative_prompt,
            base_img,
            args.controlnet_str,
            args.cfg_scale,
            args.steps
        )

    label = "".join([f"{ingredient}, " for ingredient in ingredients])
    draw_img = ImageDraw.Draw(img)
    draw_img.text((50,50),label, fill=(0,0,0), font=font)
    
    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{i:4d}.jpg", out_img)

all_files = os.listdir(args.output_dir)
all_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

selected_files = random.sample(all_files, 10)
imgs = [ cv2.imread(os.path.join(args.output_dir, file)) for file in selected_files]

row_0 = np.hstack(imgs[:5])
row_1 = np.hstack(imgs[5:])

grid = np.vstack([row_0,row_1])
print(grid.shape)
# grid = cv2.cvtColor(np.uint8(grid), cv2.COLOR_BGR2RGB)
cv2.imwrite("{args.output_dir}/grid.jpg",grid)





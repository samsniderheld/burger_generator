"""
Generate_SDXL.py

A script for doing bulk burger generation.
    - generates variable number of samples
        - samples are 2:1 random standard to random ingredients
            - uses extra patty template if extra patty in random ingredients.
        - totally random ingredients
    - stores them in a provided directory
    - saves all sample parameters in a json file.
"""

import os
import random
import time
import cv2
import numpy as np
import random
import json

from arg_parser import parse_args
from pipelines.pipelines import InpaintingSDXLPipeline

from utils.basic_utils import (load_img_for_sdxl, 
                        read_ingredients_from_txt,
                        add_text_to_image)

from utils.burger_gen_utils import (enforce_standard_ingredient_ratio,
                   contstruct_prompt_from_ingredient_list,
                   construct_negative_prompt_for_standard_ingredients)

# Parse arguments from command line or script input
args = parse_args()

random_ingredients = read_ingredients_from_txt(args.food_list)

standard_ingredients = ["lettuce", "tomatoes", "pickles", "onions", 
                        "ketchup", "cheese", "bacon", "extra patty", 
                        "mayonaise"]

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

sdxl_pipe  = InpaintingSDXLPipeline(args.sdxl_model)

all_samples = []

for i in range(args.num_samples):

    start_time = time.time()

    num_ingredients = random.randint(3,8)

    #construct prompts
    if (random.random()>.5):

        """constuct a prompt that usings standard burger king ingredients.
        If on of the ingredients is 'extra patty' then use that specific template."""
       
        #random ingredients but force a 2:1 standard to random ingredient ratio.
        ingredients = enforce_standard_ingredient_ratio(random_ingredients,
                        standard_ingredients,num_ingredients)
        
        if "extra patty" in ingredients:
          
            ingredients.remove("extra patty")

            mask_num = 10

            prompt = contstruct_prompt_from_ingredient_list(ingredients)

            negative_prompt = construct_negative_prompt_for_standard_ingredients(ingredients, standard_ingredients)

            ingredients.append("extra patty")

        else:

            mask_num = num_ingredients

            prompt = contstruct_prompt_from_ingredient_list(ingredients)

            negative_prompt = construct_negative_prompt_for_standard_ingredients(ingredients, standard_ingredients)
      
    else:

        #totally random ingredient generation
        ingredients = random.sample(random_ingredients, num_ingredients)

        prompt = contstruct_prompt_from_ingredient_list(ingredients)
        
        negative_prompt = args.negative_prompt

        mask_num = num_ingredients

    #load image and mask for inpainting
    path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
    base_img = load_img_for_sdxl(path)

    mask_path = os.path.join(args.template_dir,f"{mask_num}_ingredient_mask.png")
    mask_img = load_img_for_sdxl(mask_path)

    #generate image
    img, seed = sdxl_pipe.generate_img(
        prompt, 
        negative_prompt,
        base_img,
        mask_img,
        .95,
        args.cfg_scale,
        args.steps,
        True
    )

    sample_details = {
      "prompt": prompt, 
      "negative_prompt": negative_prompt, 
      "cfg": args.cfg_scale,
      "steps": args.steps,
      "seed": seed,
      "name": f"{i:4d}.jpg"
    }

    all_samples.append(sample_details)

    #create label and save
    label = "".join([f"{ingredient}, " for ingredient in ingredients])
    add_text_to_image(img,label,"assets/OpenSans-Regular.ttf",20)

    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    out_path = f"{args.output_dir}/{i:04d}.jpg"
    cv2.imwrite(out_path, out_img)


samples_json = {"samples":all_samples}
with open(f"{args.output_dir}/samples.json", "w") as outfile:
    json.dump(samples_json, outfile)



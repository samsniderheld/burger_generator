import os
import random
import time
import cv2
import numpy as np
import random
import json
from PIL import  ImageDraw, ImageFont

from itertools import combinations


from arg_parser import parse_sdxl_args
from pipelines.pipelines import (InpaintingSDXLPipeline)
from utils import (load_img_for_sdxl, read_ingredients_from_txt,
                   enforce_standard_ingredient_ratio,
                   contstruct_prompt_from_ingredient_list,
                   construct_negative_prompt_for_standard_ingredients)

font = ImageFont.truetype("OpenSans-Regular.ttf", 20)

# Parse arguments from command line or script input
args = parse_sdxl_args()




standard_ingredients = ["lettuce", "tomatoes", "pickles", "onions", 
                            "ketchup", "cheese", "bacon", "extra patty", 
                            "mayonaise"]

all_combinations = []
# for r in range(3, 5):
#     all_combinations.extend(list(combinations(standard_ingredients, r)))
all_combinations.extend(list(combinations(standard_ingredients, 4)))

print(f"there are {len(all_combinations)} possible combinations")
# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

sdxl_pipe  = InpaintingSDXLPipeline(args.sdxl_model)

all_samples = []

for i,ingredients in enumerate(all_combinations):

  for j in range(args.num_samples):

    start_time = time.time()

    num_ingredients = len(ingredients)

    prompt = contstruct_prompt_from_ingredient_list(ingredients)

        
    negative_prompt = construct_negative_prompt_for_standard_ingredients(ingredients, standard_ingredients)
    #load image and mask for inpainting
    if (num_ingredients<5):

        mask_num = num_ingredients
    else:
        mask_num = 7

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
        1,
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
    draw_img = ImageDraw.Draw(img)
    draw_img.text((50,50),label, fill=(0,0,0), font=font)
    
    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    ing_string = "_".join(ingredients) 
    cv2.imwrite(f"{args.output_dir}/{i:03d}_{j:02d}_{ing_string}.jpg", out_img)


samples_json = {"samples":all_samples}
with open(f"{args.output_dir}/samples.json", "w") as outfile:
    json.dump(samples_json, outfile)



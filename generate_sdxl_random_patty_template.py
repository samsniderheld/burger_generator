import os
import random
import time
import cv2
import numpy as np
import random
import json
from PIL import  ImageDraw, ImageFont

from arg_parser import parse_sdxl_args
from pipelines.pipelines import (InpaintingSDXLPipeline)
from utils import (load_img_for_sdxl, read_ingredients_from_txt,
                   enforce_standard_ingredient_ratio,
                   contstruct_prompt_from_ingredient_list,
                   construct_negative_prompt_for_standard_ingredients)

font = ImageFont.truetype("OpenSans-Regular.ttf", 20)

# Parse arguments from command line or script input
args = parse_sdxl_args()

# load files and setup directory
if(args.create_grid and args.num_samples<10):

    raise Exception("the image grid format currently only generates  5x2 images.")


random_ingredients = read_ingredients_from_txt("food_list.txt")


standard_ingredients = ["lettuce", "tomatoes", "pickles", "onions", 
                            "ketchup", "cheese", "bacon", "extra patty", 
                            "mayonaise"]

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

sdxl_pipe  = InpaintingSDXLPipeline(args.sdxl_model)

all_samples = []

for i in range(args.num_samples):

    num_ingredients = random.randint(3,8)

    start_time = time.time()

    #construct prompts
    if (random.random()>.5):

        ingredients = enforce_standard_ingredient_ratio(random_ingredients,
                        standard_ingredients,num_ingredients)
        
        if "extra patty" in ingredients:
          print("using_template")
          ingredients.remove("extra patty")

          mask_num = 8

          prompt = contstruct_prompt_from_ingredient_list(ingredients)
        
          negative_prompt = construct_negative_prompt_for_standard_ingredients(ingredients, standard_ingredients)
      
          ingredients.append("extra patty")
        else:
                #load image and mask for inpainting
          if (num_ingredients<5):

              mask_num = num_ingredients
          else:
              mask_num = 7

          prompt = contstruct_prompt_from_ingredient_list(ingredients)
          
          negative_prompt = construct_negative_prompt_for_standard_ingredients(ingredients, standard_ingredients)
      
          ingredients.append("extra patty")
    else:

        ingredients = random.sample(random_ingredients, num_ingredients)

        prompt = contstruct_prompt_from_ingredient_list(ingredients)
        
        negative_prompt = args.negative_prompt


        #load image and mask for inpainting
        if (num_ingredients<5):

            mask_num = num_ingredients
        else:
            mask_num = 7

    path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
    base_img = load_img_for_sdxl(path)

    mask_path = os.path.join(args.template_dir,f"{mask_num}_ingredient_mask.png")
    mask_img = load_img_for_sdxl(mask_path)

    print(prompt)
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
    draw_img = ImageDraw.Draw(img)
    draw_img.text((50,50),label, fill=(0,0,0), font=font)
    
    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{i:4d}.jpg", out_img)

#generate grid
if(args.create_grid):

    all_files = os.listdir(args.output_dir)
   
    all_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    selected_files = random.sample(all_files, 10)
    imgs = [ cv2.imread(os.path.join(args.output_dir, file)) for file in selected_files]

    row_0 = np.hstack(imgs[:5])
    row_1 = np.hstack(imgs[5:])

    grid = np.vstack([row_0,row_1])
    print(grid.shape)
    cv2.imwrite(f"{args.output_dir}/grid.jpg",grid)

samples_json = {"samples":all_samples}
with open(f"{args.output_dir}/samples.json", "w") as outfile:
    json.dump(samples_json, outfile)



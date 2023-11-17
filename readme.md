# README Documentation for Burger Generator

## Overview

This repo is an R&D directory designed to generate large numbers of burger images via SDXL and inpainting.

`Generate_SDXL.py` is a Python script designed to demonstrate all the elements of the repo. It combines various ingredients, both standard and random, to create a diverse set of burger samples. The script is capable of enforcing a specific ratio of standard to random ingredients and can incorporate special templates for specific ingredients like an extra patty.

## Features
- **Variable Sample Generation:** Generates a user-defined number of burger samples.
- **Ingredient Customization:** Supports a mix of standard and random ingredients, with an enforced 2:1 ratio.
- **Special Templates:** Uses a unique template for burgers with an extra patty.
- **Storage and Tracking:** Saves generated burger images in a specified directory and records their parameters in a JSON file.

## Requirements
- accelerate==0.24.1
- bitsandbytes==0.41.2.post2
- compel==2.0.2
- diffusers==0.23.0
- omegaconf==2.3.0
- safetensors==0.4.0
- torch==2.1.0+cu118
- transformers==4.35.2
- xformers==0.0.22.post7

## installation
```
pip install -r requirements.txt
```

## Example Usage
```python
import os
import random
from pipelines.pipelines import InpaintingSDXLPipeline
from utils.basic_utils import load_img_for_sdxl, read_ingredients_from_txt
from utils.burger_gen_utils import contstruct_prompt_from_ingredient_list

sdxl_pipe  = InpaintingSDXLPipeline("model_path_here")

num_ingredients = random.randint(3,8)

random_ingredients = read_ingredients_from_txt("assets/food_list.txt")

ingredients = random.sample(random_ingredients, num_ingredients)

prompt = contstruct_prompt_from_ingredient_list(ingredients)

negative_prompt = "poor quality"

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
    7,
    50,
    True
)
```

## Example Command
To run the script, use a command in the following format (assuming all required arguments are defined in `arg_parser`):
```
python Generate_SDXL.py --num_samples 50 --output_dir ./output --food_list ./ingredients.txt
```

## Output
- Generated burger images will be saved in the specified output directory.
- A JSON file (`samples.json`) containing details of all samples will also be saved in this directory.

## Note
Ensure all dependencies are installed and paths to templates, ingredient lists, and other resources are correctly set in the arguments.
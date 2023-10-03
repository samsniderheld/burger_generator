"""WIP multi ingredient gen file"""
import argparse
import os
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch

from diffusers.utils import load_image

from pipelines.pipelines import get_control_net_pipe, get_img2img_pipe
from utils import(blend_image, composite_ingredients, 
                  generate_template_and_mask,read_ingredients_from_csv)

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating burgers with multiple ingredients"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--texture_dir', type=str, default='input_templates/00.jpg', 
        help='The directory for input data')
    parser.add_argument(
        '--overlay_dir', type=str, default='burger_templates/', 
        help='The burger overlay')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_img2img_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using for img2img')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-scribble', 
        help='The controlnet model we are using.')
    parser.add_argument(
        '--steps', type=int, default=20, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_burgers', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--img2img_strength', type=float, default=.2, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--mask_blur', type=int, default=3, 
        help='How to blur mask composition')
    parser.add_argument(
        '--cfg_scale', type=float, default=3.5, 
        help='How much creativity the pipeline has')

    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

height = args.dims
width = args.dims
img2img_strength = args.img2img_strength
steps = args.steps
cfg = args.cfg_scale

img2img_pipe, img2img_proc = get_img2img_pipe(args.base_img2img_model)

negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

negative_img2img_embeds = img2img_proc(negative_prompt)

directory_path = args.texture_dir

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

file_urls = sorted(
    [
        os.path.join(directory_path, f) for f in os.listdir(directory_path) 
        if os.path.isfile(os.path.join(directory_path, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ],
    key=str.casefold  # This will sort the URLs in a case-insensitive manner
)
all_ingredients = [os.path.basename(file).split("_")[0] for file in file_urls]

all_ingredients = list(set(all_ingredients))

for i in range(args.num_burgers):

    num_ingredients = random.randint(3,6)
    ingredients = random.choices(all_ingredients, k=num_ingredients)

    overlay_top = Image.open(os.path.join(args.overlay_dir,"top.png")).convert("RGBA")
    overlay_bottom = Image.open(os.path.join(args.overlay_dir,"bottom.png")).convert("RGBA")
    

    for j in range(args.num_samples):

        start_time = time.time()
        
        burger_ingredient_string = "".join([f"{ingredient}++, " for ingredient in ingredients]) 

        burger_prompt = f"""image a burger with a {burger_ingredient_string} photorealistic photography, 8k uhd, full framed, photorealistic photography, dslr, soft lighting, high quality, Fujifilm XT3\n\n"""

        print(burger_prompt)
        print(img2img_strength,steps,cfg)

        img2img_embeds = img2img_proc(burger_prompt)    
        
        random_seed = random.randrange(0,100000)

        textures = []

        for ingredient in ingredients:

            texture_paths = [path for path in file_urls if ingredient in  path]

            texture_path = random.choice(texture_paths)

            texture = Image.open(texture_path)

            textures.append(texture)
        
        template,template_values,mask = generate_template_and_mask(len(textures),overlay_top, overlay_bottom)

        input_img = composite_ingredients(textures[::-1],template,template_values)

        
        img = img2img_pipe(prompt_embeds=img2img_embeds,
                        negative_prompt_embeds = negative_img2img_embeds,
                        image= input_img,
                        strength = img2img_strength,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]
        
        img = blend_image(img,input_img,mask,args.mask_blur)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 

        out_img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{j:4d}.jpg", out_img)


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

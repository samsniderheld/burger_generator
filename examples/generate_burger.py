"""
Burger Generator using Hugging Face Diffusers Pipeline

This script provides a pipeline for generating burger images using the Hugging Face Diffusers Pipeline.
It utilizes a set of provided models, arguments, and image processing techniques to generate 
realistic ingredient textures in a burger representation. The generated images are saved 
to the specified output directory.

"""

import sys
sys.path.append('../')
import argparse
import os
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch

from pipelines.pipelines import get_img2img_pipe
from utils import overlay_images

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating ingredient textures"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--texture', type=str, default='texture_outputs/0000.jpg', 
        help='The background texture')
    parser.add_argument(
        '--template', type=str, default='burger_templates/burger_template.png', 
        help='The burger template')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--ingredient', type=str, default='avocado', 
        help='The ingredient texture we want to generate.')
    parser.add_argument(
        '--steps', type=int, default=20, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--img2img_strength', type=float, default=.4, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--cfg_scale', type=float, default=3.5, 
        help='How much creativity the pipeline has')

    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)



img2img_strength = args.img2img_strength
steps = args.steps
cfg = args.cfg_scale

img2img_pipe, compel_proc = get_img2img_pipe(args.base_model)
    # Open the background and overlay images
texture = Image.open(args.texture).convert("RGBA")
template = Image.open(args.template).convert("RGBA")
input_img = overlay_images(texture,template)


prompt = f"""image a burger with  {args.ingredient}+++, photorealistic photography, 
8k uhd, full framed, photorealistic photography, 8k uhd, dslr, soft lighting, 
high quality, Fujifilm XT3\n\n"""

prompt_embeds = compel_proc(prompt)

negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

negative_prompt_embeds = compel_proc(negative_prompt)

for i in range(args.num_samples):

    start_time = time.time()
    
    random_seed = random.randrange(0,100000)

    img = img2img_pipe(prompt_embeds=prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    image= input_img,
                    strength = img2img_strength,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    out_img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{i:4d}.jpg", out_img)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

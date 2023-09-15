
import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from diffusers.utils import load_image

from pipelines.pipelines import get_control_net_pipe
from utils import read_ingredients_from_csv

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
        '--input_texture', type=str, default='input_templates/00.jpg', 
        help='The directory for input data')
    parser.add_argument(
        '--output_dir', type=str, default='texture_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-scribble', 
        help='The controlnet model we are using.')
    parser.add_argument(
        '--ingredient', type=str, default='avocado', 
        help='The ingredient texture we want to generate.')
    parser.add_argument(
        '--csv_file', type=str, default=None, 
        help='The ingredient texture we want to generate.')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--steps', type=int, default=20, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--controlnet_str', type=float, default=.85, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--cfg_scale', type=float, default=3.5, 
        help='How much creativity the pipeline has')

    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)



controlnet_conditioning_scale = args.controlnet_str
height = args.dims
width = args.dims
steps = args.steps
cfg = args.cfg_scale

control_net_pipe, compel_proc = get_control_net_pipe(args.controlnet_path,args.base_model)
control_net_img = load_image(args.input_texture)


if(not args.csv_file):

    prompt = f"""a 2D texture of {args.ingredient}+++, layered++, side view, 
    full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
    Fujifilm XT3\n\n"""

    prompt_embeds = compel_proc(prompt)

    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

    negative_prompt_embeds = compel_proc(negative_prompt)

    for i in range(args.num_samples):

        start_time = time.time()
        
        random_seed = random.randrange(0,100000)

        image = control_net_pipe(prompt_embeds=prompt_embeds,
                        negative_prompt_embeds = negative_prompt_embeds,
                        image= control_net_img,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height=height,
                        width=width,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        out_img = cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{args.output_dir}/{args.ingredient}/{i:04d}.jpg", out_img)

else:

    ingredients = read_ingredients_from_csv(args.csv_file)

    for ingredient in ingredients:

        print(ingredient)

        prompt = f"""a 2D texture of {ingredient}+++, layered++, side view, 
        full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
        Fujifilm XT3\n\n"""

        prompt_embeds = compel_proc(prompt)

        negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

        negative_prompt_embeds = compel_proc(negative_prompt)

        for i in range(args.num_samples):

            start_time = time.time()
            
            random_seed = random.randrange(0,100000)

            image = control_net_pipe(prompt_embeds=prompt_embeds,
                            negative_prompt_embeds = negative_prompt_embeds,
                            image= control_net_img,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            height=height,
                            width=width,
                            num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                            guidance_scale = cfg).images[0]

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The script took {elapsed_time:.2f} seconds to execute.")

            out_img = cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{args.output_dir}/{ingredient}_{i:04d}.jpg", out_img)



if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

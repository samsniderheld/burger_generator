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
from utils import blend_image, composite_ingredients, generate_template

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
        '--input_texture', type=str, default='input_templates/00.jpg', 
        help='The directory for input data')
    parser.add_argument(
        '--mask_1', type=str, default='burger_templates/burger_template_top_multi_ingredient', 
        help='The burger mask')
    parser.add_argument(
        '--mask_2', type=str, default='burger_templates/burger_template_bottom_multi_ingredient', 
        help='The burger mask')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_texture_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--base_img2img_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using for img2img')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-scribble', 
        help='The controlnet model we are using.')
    parser.add_argument(
        '--ingredients', type=str, nargs='+', default=['fried chicken', 'raspberries'],
        help='The ingredients we are generating')
    parser.add_argument(
        '--steps', type=int, default=20, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--controlnet_str', type=float, default=.85, 
        help='How much impact does the control net have.')
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

ingredients = args.ingredients

controlnet_conditioning_scale = args.controlnet_str
height = args.dims
width = args.dims
img2img_strength = args.img2img_strength
steps = args.steps
cfg = args.cfg_scale

control_net_pipe, control_proc = get_control_net_pipe(args.controlnet_path,args.base_texture_model)
control_net_img = load_image(args.input_texture)

img2img_pipe, img2img_proc = get_img2img_pipe(args.base_img2img_model)

mask_1 = Image.open(args.mask_1).convert("RGBA").resize((512,512))
mask_2 = Image.open(args.mask_2).convert("RGBA").resize((512,512))

template,template_values = generate_template(len(ingredients),mask_1,mask_2)

template_values = template_values[:-1]
ingredient_prompt_embeds = []

for ingredient in ingredients:

    prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
    full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
    Fujifilm XT3\n\n"""

    embeds = control_proc(prompt)

    ingredient_prompt_embeds.append(embeds)


burger_ingredient_string = "".join([f"{ingredient}+++," for ingredient in ingredients[1:-1]]) 

burger_prompt = f"""image a burger with {burger_ingredient_string}, photorealistic photography, 
8k uhd, full framed, photorealistic photography, dslr, soft lighting, 
high quality, Fujifilm XT3\n\n"""

print(burger_prompt)

img2img_embeds = img2img_proc(burger_prompt)

negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

negative_control_embeds = control_proc(negative_prompt)
negative_img2img_embeds = img2img_proc(negative_prompt)

for i in range(args.num_samples):

    start_time = time.time()
    
    random_seed = random.randrange(0,100000)

    textures = []

    for embeds in ingredient_prompt_embeds:

        texture = control_net_pipe(prompt_embeds=embeds,
                        negative_prompt_embeds = negative_control_embeds,
                        image= control_net_img,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height=height,
                        width=width,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]
        
        textures.append(texture)
    
    
    input_img = composite_ingredients(textures,template,template_values)

    cv2.imwrite(f"composite.jpg", cv2.cvtColor(np.uint8(input_img),cv2.COLOR_BGR2RGB))
    img = img2img_pipe(prompt_embeds=img2img_embeds,
                    negative_prompt_embeds = negative_img2img_embeds,
                    image= input_img,
                    strength = img2img_strength,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    
    # img = blend_image(img,input_img,combined_mask,args.mask_blur)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients[1:-1]]) 

    out_img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)

    pipeline_img = np.hstack([template,input_img, img.convert('RGB')])
    pipeline_img = cv2.cvtColor(np.uint8(pipeline_img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"pipeline_img.jpg", pipeline_img)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

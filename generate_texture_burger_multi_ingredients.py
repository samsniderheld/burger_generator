"""WIP multi ingredient gen file"""
import argparse
import os
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch


from pipelines.pipelines import (get_control_net_pipe, get_img2img_pipe,
ControlNetPipeline,Img2ImgPipeline)
from utils import(blend_image, composite_ingredients, 
                  generate_template_and_mask,read_ingredients_from_txt)

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
        '--overlay_dir', type=str, default='burger_templates/', 
        help='The burger overlay')
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
    parser.add_argument(
        '--txt_file', type=str, default=None, 
        help='The ingredient texture we want to generate.')

    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

controlnet_conditioning_scale = args.controlnet_str
height = args.dims
width = args.dims
img2img_strength = args.img2img_strength
steps = args.steps
cfg = args.cfg_scale

# control_net_pipe, control_proc = get_control_net_pipe(args.controlnet_path,args.base_texture_model)

controlnet_pipe = ControlNetPipeline(args.base_texture_model, args.controlnet_path)
control_net_img = Image.open(args.input_texture)

img2img_pipe = Img2ImgPipeline(args.base_img2img_model)
# img2img_pipe, img2img_proc = get_img2img_pipe(args.base_img2img_model)

negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

overlay_top = Image.open(os.path.join(args.overlay_dir,"top.png")).convert("RGBA")
overlay_bottom = Image.open(os.path.join(args.overlay_dir,"bottom.png")).convert("RGBA")


for i in range(args.num_samples):

    start_time = time.time()

    #set up ingredient list
    ingredient_prompt_embeds = []

    if args.txt_file != None:
        all_ingredients = read_ingredients_from_txt(args.txt_file)
        ingredients = random.choices(all_ingredients,k=random.randint(2,5))

    else:
        ingredients = args.ingredients

    #create templates, values, and mask
    template,template_values,mask = generate_template_and_mask(len(ingredients), overlay_top, overlay_bottom)

    #generated ingredient texture/s
    textures = []
    for ingredient in ingredients:

        prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
        full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
        Fujifilm XT3\n\n"""

        texture = controlnet_pipe.generate_img(prompt,
                        negative_prompt,
                        control_net_img,
                        controlnet_conditioning_scale,
                        width,
                        height,
                        steps,
                        cfg)
        
        textures.append(texture)
        
    burger_ingredient_string = "".join([f"{ingredient}, " for ingredient in ingredients]) 

    burger_prompt = f"""image a burger with a burger king beef patty+++, {burger_ingredient_string}, poppyseed bun+++, photorealistic photography, 
    8k uhd, full framed, photorealistic photography, dslr, soft lighting, 
    high quality, Fujifilm XT3\n\n"""

    print(burger_prompt)

    
    input_img = composite_ingredients(textures[::-1],template,template_values)

    img = img2img_pipe(burger_prompt,
                    negative_prompt,
                    input_img,
                    img2img_strength,
                    steps,
                    cfg)
    
    img = blend_image(img,input_img,mask,args.mask_blur)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 

    out_img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)

    pipeline_img = np.hstack([template,input_img, img.convert('RGB')])
    pipeline_img = cv2.cvtColor(np.uint8(pipeline_img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"pipeline_img.jpg", pipeline_img)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

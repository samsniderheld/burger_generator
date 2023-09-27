"""
gradio_interface.py

This module provides an interactive Gradio interface to demonstrate and operate
the Hugging Face Diffusers Pipeline for generating ingredient textures.

Functions:
    - parse_args() -> argparse.Namespace: 
        Parse command-line arguments for setting up and running the pipeline.
        
    - generate_texture(ingredient, controlnet_img, controlnet_conditioning_scale, steps, cfg) -> numpy.ndarray:
        Generate the texture for a given ingredient using the control net pipeline.

    - generate_burger(ingredient, strength, mask_blur_strength, steps, cfg) -> PIL.Image:
        Generate a burger image with the ingredient texture applied.
        
    - Gradio Interface:
        Create an interactive Gradio dashboard to allow users to generate textures and burgers
        with different ingredients and settings.

Usage:
    To be executed as a standalone script, and it will launch a Gradio dashboard in a web browser.

Example:
    $ python gradio_interface.py --input_texture "path_to_texture.jpg" --ingredient "avocado"
"""

import sys
# sys.path.append('../')
import argparse
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

from diffusers.utils import load_image

from pipelines.pipelines import get_control_net_pipe, get_img2img_pipe
from utils import (overlay_images, blend_image, 
                   generate_template_and_mask, composite_ingredients)

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
        '--template', type=str, default='burger_templates/burger_template.png', 
        help='The burger template')
    parser.add_argument(
        '--overlay_dir', type=str, default='burger_templates/', 
        help='The burger overlay')
    parser.add_argument(
        '--mask', type=str, default='burger_templates/burger_mask.png', 
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
        '--ingredient', type=str, default='avocado', 
        help='The ingredient texture we want to generate.')
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

os.makedirs("temp_textures", exist_ok=True)
control_net_pipe, control_proc = get_control_net_pipe(args.controlnet_path,args.base_texture_model)
control_net_img = load_image(args.input_texture)

img2img_pipe, img2img_proc = get_img2img_pipe(args.base_img2img_model)

all_textures = []
template = None
composite = None
mask = None
all_ingredients = []

def generate_texture(ingredients,controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates the texture
    global all_textures, all_ingredients
    all_textures = []

    start_time = time.time()

    all_ingredients = ingredients.split(",")

    for ingredient in all_ingredients:

        texture_prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
        full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
        Fujifilm XT3\n\n"""

        control_embeds = control_proc(texture_prompt)

        negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

        negative_control_embeds = control_proc(negative_prompt)

        start_time = time.time()
        
        random_seed = random.randrange(0,100000)

        controlnet_input = Image.fromarray(controlnet_img)

        texture_img = control_net_pipe(prompt_embeds=control_embeds,
                        negative_prompt_embeds = negative_control_embeds,
                        image= controlnet_input,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height=512,
                        width=512,
                        num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                        guidance_scale = cfg).images[0]
        
        all_textures.append(texture_img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    return all_textures

def generate_template():
    global all_textures,template,mask,composite

    overlay = Image.open(os.path.join(args.overlay_dir,
                                  f"{len(all_textures)}_ingredient.png")).convert("RGBA").resize((512,512))
    template,template_values,mask = generate_template_and_mask(len(all_textures),overlay)

    composite = composite_ingredients(all_textures[::-1],template,template_values)

    return [template, mask,composite]


def generate_burger(strength,mask_blur_strength,steps,cfg):

    global all_textures,template,mask,composite,all_ingredients

    start_time = time.time()

    burger_ingredient_string = "".join([f"{ingredient}, " for ingredient in all_ingredients]) 

    burger_prompt = f"""image a burger with {ingredient}+++, photorealistic photography, 
    8k uhd, full framed, photorealistic photography, dslr, soft lighting, 
    high quality, Fujifilm XT3\n\n"""

    img2img_embeds = img2img_proc(burger_prompt)

    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

    negative_img2img_embeds = img2img_proc(negative_prompt)

    start_time = time.time()
    
    random_seed = random.randrange(0,100000)

     
    img = img2img_pipe(prompt_embeds=img2img_embeds,
                    negative_prompt_embeds = negative_img2img_embeds,
                    img= composite,
                    strength = strength,
                    num_inference_steps=steps, 
                    generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    
    img = blend_image(img,composite,mask,mask_blur_strength)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    return img
    


with gr.Blocks() as demo:

    with gr.Tab("Input Textures"):
        #texture gen row
        with gr.Row():

                with gr.Column():
                    controlnet_prompt_input = gr.Textbox(label="prompt")
                    controlnet_input_img = gr.Image(label="input img")
                    controlnet_conditioning_scale_input = gr.Slider(0, 1, 
                        value=args.controlnet_str, label="controlnet_conditioning_scale")
                    controlnet_steps_input = gr.Slider(0, 150, value=args.steps,
                        label="number of diffusion steps")
                    controlnet_cfg_input = gr.Slider(0,30,value=args.cfg_scale,label="cfg scale")

                    controlnet_inputs = [
                        controlnet_prompt_input,
                        controlnet_input_img,
                        controlnet_conditioning_scale_input,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                    ]

                with gr.Column():

                    controlnet_output = gr.Gallery()

        with gr.Row():

                controlnet_submit = gr.Button("Submit")
    with gr.Tab("Templates"):
         template_ouput = gr.Gallery()
         template_submit = gr.Button("Submit")

    with gr.Tab("Burger Gen"):
        #burger gen row
        with gr.Row():

                with gr.Column():
                    img2img_strength = gr.Slider(0, 1, 
                        value=args.img2img_strength, 
                        label="img2img strength")
                    mask_blur_strength = gr.Slider(0, 9, 
                        value=args.mask_blur, 
                        label="mask blur")
                    steps_input = gr.Slider(0, 150, value=args.steps,
                        label="number of diffusion steps")
                    img2img_cfg_input = gr.Slider(0,30,value=args.cfg_scale,
                                                    label="cfg scale")

                    img2img_inputs = [
                        #use the same prompt from step 1
                        img2img_strength,
                        mask_blur_strength,
                        steps_input,
                        img2img_cfg_input,
                    ]

                with gr.Column():

                    img2img_output = gr.Image()

        with gr.Row():

                img2img_submit = gr.Button("Submit")


    controlnet_submit.click(generate_texture,inputs=controlnet_inputs,outputs=controlnet_output)
    template_submit.click(generate_template,outputs=template_ouput)
    img2img_submit.click(generate_burger,inputs=img2img_inputs,outputs=img2img_output)


if __name__ == "__main__":
    demo.launch(share=True,debug=True)
"""
compare_models.py

This module provides an interactive way to compare the outputs of two models in generating ingredient 
textures based on user-defined input. 
"""

import sys
sys.path.append('../')
import argparse
import os
import random

import gradio as gr
import numpy as np
from PIL import Image
import torch

from pipelines.pipelines import get_control_net_pipe, get_img2img_pipe

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
        '--model_1', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The first model to test')
    parser.add_argument(
        '--model_2', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='the second model to test')
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
        '--cfg_scale', type=float, default=3.5, 
        help='How much creativity the pipeline has')

    return parser.parse_args()

args = parse_args()

os.makedirs("temp_textures", exist_ok=True)
model_1_pipe, model_1_proc = get_control_net_pipe(args.controlnet_path,args.model_1)
model_2_pipe, model_2_proc = get_control_net_pipe(args.controlnet_path,args.model_2)



def generate_texture(ingredient,controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates the texture

    texture_prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
    full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
    Fujifilm XT3\n\n"""

    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

    model_1_embeds = model_1_proc(texture_prompt)

    negative_control_embeds_1 = model_1_proc(negative_prompt)

    model_2_embeds = model_2_proc(texture_prompt)

    negative_control_embeds_2 = model_2_proc(negative_prompt)

    
    random_seed = random.randrange(0,100000)

    controlnet_img = Image.fromarray(controlnet_img)

    texture_1 = model_1_pipe(prompt_embeds=model_1_embeds,
                    negative_prompt_embeds = negative_control_embeds_1,
                    image= controlnet_img,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    height=512,
                    width=512,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    
    texture_2 = model_2_pipe(prompt_embeds=model_2_embeds,
                    negative_prompt_embeds = negative_control_embeds_2,
                    image= controlnet_img,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    height=512,
                    width=512,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    

    test_img = np.hstack([texture_1,texture_2])
    
    return test_img
    
with gr.Blocks() as demo:

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

                controlnet_output = gr.Image()

    with gr.Row():

            controlnet_submit = gr.Button("Submit")



    controlnet_submit.click(generate_texture,inputs=controlnet_inputs,outputs=controlnet_output)

demo.launch(share=True,debug=True)
import argparse
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
import torch

from diffusers.utils import load_image

from pipelines.pipelines import get_control_net_pipe, get_img2img_pipe
from utils import overlay_images, blend_image

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

def generate_texture(ingredient,controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates the texture

    start_time = time.time()

    texture_prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
    full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
    Fujifilm XT3\n\n"""

    control_embeds = control_proc(texture_prompt)

    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

    negative_control_embeds = control_proc(negative_prompt)

    start_time = time.time()
    
    random_seed = random.randrange(0,100000)

    controlnet_img = Image.fromarray(controlnet_img)

    texture_image = control_net_pipe(prompt_embeds=control_embeds,
                    negative_prompt_embeds = negative_control_embeds,
                    image= controlnet_img,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    height=512,
                    width=512,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    out_img = cv2.cvtColor(np.uint8(texture_image),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"temp_textures/texture.jpg", out_img)
    
    return texture_image

def generate_burger(ingredient,strength,steps,cfg):

    start_time = time.time()

    burger_prompt = f"""image a burger with {ingredient}+++, photorealistic photography, 
    8k uhd, full framed, photorealistic photography, dslr, soft lighting, 
    high quality, Fujifilm XT3\n\n"""

    img2img_embeds = img2img_proc(texture_prompt)

    negative_prompt = f'illustration, sketch, drawing, poor quality, low quality'

    negative_img2img_embeds = img2img_proc(negative_prompt)

    start_time = time.time()
    
    random_seed = random.randrange(0,100000)

    texture = Image.open("temp_textures/texture.jpg").convert("RGBA")
    template = Image.open(args.template).convert("RGBA")
    input_img = overlay_images(texture,template)
     
    image = img2img_pipe(prompt_embeds=img2img_embeds,
                    negative_prompt_embeds = negative_img2img_embeds,
                    image= input_img,
                    strength = img2img_strength,
                    num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(random_seed),
                    guidance_scale = cfg).images[0]
    
    image = blend_image(image,input_img,mask,args.mask_blur)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    return image
    

    #generates the burger

# def save_config():
#     #saves the config

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

    #burger gen row
    with gr.Row():

            with gr.Column():
                img2img_strength = gr.Slider(0, 1, 
                    value=args.img2img_strength, 
                    label="img2img strength")
                img2img_steps_input = gr.Slider(0, 150, value=args.steps,
                    label="number of diffusion steps")
                img2img_cfg_input = gr.Slider(0,30,value=args.cfg_scale,
                                                 label="cfg scale")

                img2img_inputs = [
                    #use the same prompt from step 1
                    controlnet_prompt_input,
                    img2img_strength,
                    img2img_steps_input,
                    img2img_cfg_input,
                ]

            with gr.Column():

                img2img_output = gr.Image()

    with gr.Row():

            img2img_submit = gr.Button("Submit")


    controlnet_submit.click(generate_texture,inputs=controlnet_inputs,outputs=controlnet_output)
    img2img_submit.click(generate_burger,inputs=img2img_inputs,outputs=img2img_output)


if __name__ == "__main__":
    demo.launch(share=True,debug=True)
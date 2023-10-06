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
import os
import time
import numpy as np
import cv2

import gradio as gr

from arg_parser import parse_gradio_args
from PIL import Image

from pipelines.pipelines import ControlNetPipeline,Img2ImgPipeline
from utils import (blend_image, 
                   generate_template_and_mask, composite_ingredients)

args = parse_gradio_args()

controlnet_pipe = ControlNetPipeline(args.base_texture_model, args.controlnet_path)
img2img_pipe = Img2ImgPipeline(args.base_img2img_model)

controlnet_texture = None
all_textures = []
template = None
composite = None
mask = None
all_ingredients = []
controlnet_str = None
controlnet_cfg = None
controlnet_steps = None
burger_str = None
burger_cfg = None
burger_steps = None

def generate_texture(ingredients,controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates the texture
    global all_textures, all_ingredients
    global controlnet_texture, controlnet_str, controlnet_cfg, controlnet_steps
    controlnet_str = controlnet_conditioning_scale
    controlnet_cfg = cfg
    controlnet_steps = steps
    
    all_textures = []

    start_time = time.time()

    all_ingredients = ingredients.split(",")

    for ingredient in all_ingredients:

        texture_prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
        full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
        Fujifilm XT3\n\n"""

        start_time = time.time()
        
        controlnet_input = Image.fromarray(controlnet_img)

        controlnet_texture = controlnet_input

        texture_img = controlnet_pipe.generate_img(texture_prompt,
                        controlnet_input,
                        controlnet_conditioning_scale,
                        512,
                        512,
                        steps,
                        cfg)
        
        all_textures.append(texture_img)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    return all_textures

def generate_template():
    global all_textures,template,mask,composite
    
    overlay_top = Image.open(os.path.join(args.overlay_dir,"top.png")).convert("RGBA")
    overlay_bottom = Image.open(os.path.join(args.overlay_dir,"bottom.png")).convert("RGBA")
    
    template,template_values,mask = generate_template_and_mask(len(all_textures),overlay_top, overlay_bottom)

    composite = composite_ingredients(all_textures[::-1],template,template_values)

    return [template, mask,composite]


def generate_burger(strength,mask_blur_strength,steps,cfg):
    global all_textures,template,mask,composite,all_ingredients
    global burger_str, burger_cfg, burger_steps
    global controlnet_texture,controlnet_str, controlnet_cfg, controlnet_steps

    burger_str = strength
    burger_cfg = cfg
    burger_steps = steps

    start_time = time.time()

    burger_ingredient_string = "".join([f"{ingredient}++, " for ingredient in all_ingredients]) 

    burger_prompt = f"""image a burger with a {burger_ingredient_string} photorealistic photography, 8k uhd, full framed, photorealistic photography, dslr, soft lighting, high quality, Fujifilm XT3\n\n"""

    print(burger_prompt)

    start_time = time.time()

    img = img2img_pipe.generate_img(burger_prompt,
                    composite,
                    burger_str,
                    burger_steps,
                    cfg)
    
    img = blend_image(img,composite,mask,mask_blur_strength)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    params = [f"prompt: {burger_prompt}",
              f"controlnet_str: {controlnet_str}", 
              f"controlnet_cfg: {controlnet_cfg}", 
              f"controlnet_steps: {controlnet_steps}",
              f"burger_str: {burger_str}", 
              f"burger_cfg: {burger_cfg}", 
              f"burger_steps: {steps}"
              ]
    
    with open("parameters.txt", 'w') as f:
        for line in params:
            f.write(f"{line}\n")


    control_input = np.array(controlnet_texture.resize((512,512)))
    output_images = [control_input] + all_textures + [template,composite, img.convert('RGB')]

    pipeline_img = np.hstack(output_images)
    pipeline_img = cv2.cvtColor(np.uint8(pipeline_img),cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"pipeline_img.jpg", pipeline_img)

    return [composite,img]

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

                    img2img_output = gr.Gallery()

        with gr.Row():

                img2img_submit = gr.Button("Submit")


    controlnet_submit.click(generate_texture,inputs=controlnet_inputs,outputs=controlnet_output)
    template_submit.click(generate_template,outputs=template_ouput)
    img2img_submit.click(generate_burger,inputs=img2img_inputs,outputs=img2img_output)


if __name__ == "__main__":
    demo.launch(share=True,debug=True)
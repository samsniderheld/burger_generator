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

from pipelines.pipelines import ControlNetPipeline
from utils import read_ingredients_from_txt

args = parse_gradio_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

controlnet_pipe = ControlNetPipeline(args.base_texture_model, args.controlnet_path)

controlnet_texture = None
template = None
controlnet_str = None
controlnet_cfg = None
controlnet_steps = None

all_ingredients = read_ingredients_from_txt(args.txt_file)
index = 0
current_ingredient = current_ingredient = all_ingredients[index]



def get_next():
     global current_ingredient,index,all_ingredients
     index+=1
     current_ingredient = all_ingredients[index]
     return current_ingredient

def get_previous():
     global current_ingredient,index,all_ingredients
     index-=1
     current_ingredient = all_ingredients[index]

     return current_ingredient


def generate_texture(controlnet_img,controlnet_conditioning_scale,steps,cfg):
    #generates the texture
    global current_ingredient
    global controlnet_texture, controlnet_str, controlnet_cfg, controlnet_steps
    controlnet_str = controlnet_conditioning_scale
    controlnet_cfg = cfg
    controlnet_steps = steps

    start_time = time.time()

    texture_prompt = f"""a 2D texture of food, {current_ingredient}+++, layered++, side view, 
    full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
    Fujifilm XT3\n\n"""

    print(texture_prompt)

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")

    out_img = cv2.cvtColor(np.uint8(texture_img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{current_ingredient}.jpg", out_img)

    return [texture_img]



with gr.Blocks() as demo:

    with gr.Tab("Input Textures"):
        #texture gen row
        with gr.Row():

                with gr.Column():
                    controlnet_input_img = gr.Image(label="input img")
                    controlnet_conditioning_scale_input = gr.Slider(0, 1, 
                        value=args.controlnet_str, label="controlnet_conditioning_scale")
                    controlnet_steps_input = gr.Slider(0, 150, value=args.steps,
                        label="number of diffusion steps")
                    controlnet_cfg_input = gr.Slider(0,30,value=args.cfg_scale,label="cfg scale")

                    controlnet_inputs = [
                        controlnet_input_img,
                        controlnet_conditioning_scale_input,
                        controlnet_steps_input,
                        controlnet_cfg_input,
                    ]

                with gr.Column():

                    ingredient = gr.Textbox()
                    controlnet_output = gr.Gallery()


        with gr.Row():
                previous_buttom_submit = gr.Button("previous")
                controlnet_submit = gr.Button("Submit")
                next_buttom_submit = gr.Button("next")
                


    controlnet_submit.click(generate_texture,inputs=controlnet_inputs,outputs=controlnet_output)
    next_buttom_submit.click(get_next,outputs=ingredient)
    previous_buttom_submit.click(get_previous,outputs=ingredient)


if __name__ == "__main__":
    demo.launch(share=True,debug=True)
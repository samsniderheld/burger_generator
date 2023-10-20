""""
pipelines.py

This module provides utility classes and functions to facilitate the process of image generation
using various pipelines. The pipelines interface with pre-trained models from the `diffusers` 
and `compel` modules to produce images based on provided textual prompts.

Classes:
    ControlNetPipeline: 
        A pipeline wrapper around the `StableDiffusionControlNetPipeline` 
        for image generation based on the ControlNet model.

    Img2ImgPipeline:
        A pipeline wrapper around the `StableDiffusionImg2ImgPipeline` for image-to-image 
        conversion tasks.

Usage Example:
    controlnet_pipe = ControlNetPipeline(pipeline_path, controlnet_path)
    generated_image = controlnet_pipe.generate_img(prompt, img, controlnet_str, width, height, steps, cfg)

    img2img_pipe = Img2ImgPipeline(pipeline_path)
    converted_image = img2img_pipe.generate_img(prompt, input_img, strength, steps, cfg)

"""
import random
import torch

# Import necessary modules from the diffusers package
from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLPipeline
)

# Import scheduler for controlling the pipeline's learning rate adjustments
from diffusers import UniPCMultistepScheduler

# Import Compel for tokenization and text encoding for image generation
from compel import Compel


class ControlNetPipeline():
    def __init__(self, pipeline_path,controlnet_path):
        # Store paths for the pipeline and controlnet
        self.pipeline_path = pipeline_path
        self.controlnet_path = controlnet_path
        
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self):
        # Load the pretrained ControlNet model
        controlnet = ControlNetModel.from_pretrained(self.controlnet_path)
        
        # Load the pipeline using the pretrained ControlNet model
        controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.pipeline_path,
            controlnet=controlnet,
            safety_checker=None,
        ).to('cuda')

        # Set scheduler for the pipeline
        controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
        
        # Initialize the Compel processor for tokenization and encoding
        compel_proc = Compel(tokenizer=controlnet_pipe.tokenizer, text_encoder=controlnet_pipe.text_encoder)

        # Store the loaded pipeline and Compel processor as instance attributes
        self.pipeline = controlnet_pipe
        self.compel_proc = compel_proc

    def generate_img(self,prompt, img, controlnet_str, width, height, steps, cfg):
        # Convert prompt to embeddings
        prompt_embeds = self.compel_proc(prompt)
        
        # Negative prompts to avoid certain characteristics in the generated image
        negative_prompt = 'illustration, sketch, drawing, poor quality, low quality'
        negative_prompt_embeds = self.compel_proc(negative_prompt)
        
        # Generate a random seed for reproducibility
        random_seed = random.randrange(0,100000)
        
        # Generate the image using the pipeline
        img = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=img,
            controlnet_conditioning_scale=controlnet_str,
            width=width,
            height=height,
            num_inference_steps=steps, 
            generator=torch.Generator(device='cuda').manual_seed(random_seed),
            guidance_scale=cfg).images[0]
        
        return img


class Img2ImgPipeline():
    def __init__(self, pipeline_path):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self):
        # Load the Image-to-Image pipeline
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.pipeline_path,
            safety_checker=None,
        ).to('cuda')
        
        # Initialize the Compel processor for tokenization and encoding
        compel_proc = Compel(tokenizer=img2img_pipe.tokenizer, text_encoder=img2img_pipe.text_encoder)
        
        # Store the loaded pipeline and Compel processor as instance attributes
        self.pipeline = img2img_pipe
        self.compel_proc = compel_proc
        
    def generate_img(self, prompt,input_img, strength, steps, cfg):
        # Convert prompt to embeddings
        prompt_embeds = self.compel_proc(prompt)
        
        # Negative prompts to avoid certain characteristics in the generated image
        negative_prompt = 'illustration, sketch, drawing, poor quality, low quality'
        negative_prompt_embeds = self.compel_proc(negative_prompt)
        
        # Generate a random seed for reproducibility
        random_seed = random.randrange(0,100000)
        
        # Generate the image using the pipeline
        img = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=input_img,
            strength=strength,
            num_inference_steps=steps, 
            generator=torch.Generator(device='cuda').manual_seed(random_seed),
            guidance_scale = cfg).images[0]
        
        return img
    
class SDXLPipeline():
    def __init__(self, pipeline_path):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self,load_from_file=False):

        if load_from_file:
            sdxl_pipe = StableDiffusionXLPipeline.from_single_file(
                self.pipeline_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                safety_checker=None,
            )
            sdxl_pipe = sdxl_pipe.to("cuda")
        else:
            # Load the Image-to-Image pipeline
            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
                self.pipeline_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                safety_checker=None,
            )
            sdxl_pipe = sdxl_pipe.to("cuda")
        
        # Store the loaded pipeline and Compel processor as instance attributes
        self.pipeline = sdxl_pipe
        
        
    def generate_img(self, prompt, steps, cfg):        
        # Negative prompts to avoid certain characteristics in the generated image
        negative_prompt = 'illustration, sketch, drawing, poor quality, low quality'
        # negative_prompt_embeds = self.compel_proc(negative_prompt)
        # Generate a random seed for reproducibility
        random_seed = random.randrange(0,100000)

        img = self.pipeline(
            prompt = prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps, 
            generator=torch.Generator(device='cuda').manual_seed(random_seed),
            guidance_scale = cfg).images[0]
        
        
        return img

class Img2ImgSDXLPipeline():
    def __init__(self, pipeline_path):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self):
        # Load the Image-to-Image pipeline
        img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.pipeline_path,
            safety_checker=None,
        )
        img2img_pipe.enable_model_cpu_offload()
        
        # Initialize the Compel processor for tokenization and encoding
        # compel_proc = Compel(tokenizer=img2img_pipe.tokenizer, text_encoder=img2img_pipe.text_encoder)
        
        # Store the loaded pipeline and Compel processor as instance attributes
        self.pipeline = img2img_pipe
        # self.compel_proc = compel_proc
        
    def generate_img(self, prompt,input_img, strength, steps, cfg):
        # Convert prompt to embeddings
        # prompt_embeds = self.compel_proc(prompt)
        
        # Negative prompts to avoid certain characteristics in the generated image
        negative_prompt = 'illustration, sketch, drawing, poor quality, low quality'
        # negative_prompt_embeds = self.compel_proc(negative_prompt)
        
        # Generate a random seed for reproducibility
        random_seed = random.randrange(0,100000)

        original_size = input_img.size
        input_img = input_img.resize((1024,1024))

        img = self.pipeline(
            prompt = prompt,
            negative_prompt=negative_prompt,
            image=input_img,
            strength=strength,
            num_inference_steps=steps, 
            generator=torch.Generator(device='cuda').manual_seed(random_seed),
            guidance_scale = cfg).images[0]
        
        img = img.resize(original_size)
        
        return img

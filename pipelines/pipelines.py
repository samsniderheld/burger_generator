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
import numpy as np
import torch
import cv2
from PIL import Image
# Import necessary modules from the diffusers package
from diffusers import (ControlNetModel,
                       StableDiffusionXLControlNetPipeline,
                       AutoPipelineForInpainting,
                       DPMSolverMultistepScheduler
)
 
class ControlnetSDXLPipeline():
    def __init__(self, pipeline_path):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self):
        # Load the Image-to-Image pipeline
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        )

        sdxl_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.pipeline_path,
            controlnet=controlnet,
            torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True
        ).to('cuda')

        sdxl_pipe.scheduler =  DPMSolverMultistepScheduler.from_config(sdxl_pipe.scheduler.config, use_karras=True)   
        
        self.pipeline = sdxl_pipe

    def generate_img(self, prompt,negative_prompt,base_img, controlnet_conditioning_scale, cfg,steps):
        
        base_img = np.array(base_img)
        controlnet_img = cv2.Canny(base_img, 150, 200)
        controlnet_img = controlnet_img[:, :, None]
        controlnet_img = np.concatenate([controlnet_img, controlnet_img, controlnet_img], axis=2)
        controlnet_img = Image.fromarray(controlnet_img)

        img = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_img,
            width=1024,
            height=1024,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=cfg,
            num_inference_steps=steps,
        ).images[0]
                
        return img

class InpaintingSDXLPipeline():
    def __init__(self, pipeline_path):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        # Load the pipeline upon initialization
        self.load_pipeline()

    def load_pipeline(self):
        # Load the Image-to-Image pipeline
        sdxl_pipe = AutoPipelineForInpainting.from_pretrained(
                    self.pipeline_path,
                    torch_dtype=torch.float16,
                    variant="fp16").to("cuda")

        sdxl_pipe.scheduler =  DPMSolverMultistepScheduler.from_config(sdxl_pipe.scheduler.config, use_karras=True)   
        
        self.pipeline = sdxl_pipe

    def generate_img(self, prompt,negative_prompt,input_img, mask_img, strength, cfg, steps):        

        img = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image= input_img,
            mask_image=mask_img,
            strength=strength,
            guidance_scale=cfg,
            num_inference_steps=steps,
        ).images[0]
                
        return img
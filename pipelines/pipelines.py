"""
pipelines.py

Module providing utilities for constructing pipelines from pretrained models.

This module contains functions to create three distinct pipelines:
1. Control Net Pipeline
2. Image-to-Image Pipeline

Each function returns a tuple of the pipeline and its associated COMPEL processor.

Imports:
    - diffusers: For importing the ControlNetModel, StableDiffusionControlNetPipeline,
      StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, and UniPCMultistepScheduler.
    - compel: For importing the Compel class.

Functions:
    - get_control_net_pipe(control_path, sd_path): Construct a control net pipeline.
    - get_img2img_pipe(sd_path): Construct an image-to-image pipeline.

"""
from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
)
from diffusers import UniPCMultistepScheduler

from compel import Compel

def get_control_net_pipe(control_path, sd_path):
    controlnet = ControlNetModel.from_pretrained(control_path)
    control_netpipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        safety_checker=None,
    ).to('cuda')
    control_netpipe.scheduler = UniPCMultistepScheduler.from_config(control_netpipe.scheduler.config)
    compel_proc = Compel(tokenizer=control_netpipe.tokenizer, text_encoder=control_netpipe.text_encoder)

    return control_netpipe, compel_proc

def get_img2img_pipe(sd_path):
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_path,
        safety_checker=None,
    ).to('cuda')

    compel_proc = Compel(tokenizer=img2img_pipe.tokenizer, text_encoder=img2img_pipe.text_encoder)
    return img2img_pipe, compel_proc

def get_SDXL_img2img_pipe(sd_path):
    img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        sd_path,
        safety_checker=None,
    )
    img2img_pipe.enable_model_cpu_offload()
    return img2img_pipe
from diffusers import (ControlNetModel,StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
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

def get_inpaint_pipe(sd_path):
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_path,
        safety_checker=None,
    ).to('cuda')

    compel_proc = Compel(tokenizer=inpaint_pipe.tokenizer, text_encoder=inpaint_pipe.text_encoder)
    return inpaint_pipe, compel_proc


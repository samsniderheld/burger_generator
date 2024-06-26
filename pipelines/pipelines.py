""""
pipelines.py

This module provides utility classes and functions to facilitate the process of image generation
using various pipelines. The pipelines interface with pre-trained models from the `diffusers` 
and `compel` modules to produce images based on provided textual prompts.

It's designed to make iterating to image generation loops easy.
"""
import torch
import random
from diffusers import (
    AutoPipelineForInpainting,
    DPMSolverMultistepScheduler
)

from compel import Compel, ReturnedEmbeddingsType

from utils.basic_utils import blend_image
from utils.burger_gen_utils import chunk_embeds

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput


class CLIPTextModelWithProjection():
    def __init__(self, text_encoder):
        print("using neo's code")
        self.text_encoder = text_encoder

    def __getattr__(self, attr):
        return getattr(self.text_encoder, attr)

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return self.text_encoder(input_ids.long(), **kwargs)


class InpaintingSDXLPipeline():
    def __init__(self, pipeline_path,use_compel_mod=False,use_freeU=False):
        # Store the path for the pipeline
        self.pipeline_path = pipeline_path
        # Load the pipeline upon initialization
        self.use_freeU = use_freeU
        self.use_compel_mod = use_compel_mod
        self.load_pipeline()
        

    def load_pipeline(self):
        # Load the Image-to-Image pipeline
        sdxl_pipe = AutoPipelineForInpainting.from_pretrained(
                    self.pipeline_path,
                    torch_dtype=torch.float16,
                    variant="fp16").to("cuda")

        sdxl_pipe.scheduler =  DPMSolverMultistepScheduler.from_config(
        sdxl_pipe.scheduler.config, 
        use_karras=True, 
        euler_at_final=True,
        rescale_betas_zero_snr=True, 
        timestep_spacing="trailing")

        if(self.use_freeU):
          sdxl_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)   

        self.pipeline = sdxl_pipe

        if(self.use_compel_mod):

          self.compel = Compel(
              tokenizer=[sdxl_pipe.tokenizer, sdxl_pipe.tokenizer_2],
              text_encoder=[CLIPTextModelWithProjection(sdxl_pipe.text_encoder), CLIPTextModelWithProjection(sdxl_pipe.text_encoder_2)],
              returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
              requires_pooled=[False, True],
              truncate_long_prompts=False,
          )

        
        else:
        
          self.compel = Compel(
            tokenizer=[sdxl_pipe.tokenizer, sdxl_pipe.tokenizer_2] ,
            text_encoder=[sdxl_pipe.text_encoder, sdxl_pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False
          )

    def generate_img(self, prompt,negative_prompt,input_img, mask_img, strength, cfg, steps, seed=None, use_chunking=True, use_compel_only=False, blend_img=False):        

        if(use_chunking):
            #chunk the prompt more akin to webui
            conditioning,pooled = chunk_embeds(prompt, self.pipeline, self.compel)
            negative_conditioning,negative_pooled = chunk_embeds(negative_prompt, self.pipeline, self.compel)

        elif(use_compel_only):
            conditioning, pooled = self.compel(prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)

        if seed == None:
            seed = random.randint(0,10000)

        generator = torch.Generator(device='cuda').manual_seed(seed)

        if(use_chunking == False and use_compel_only == False):
           
            img = self.pipeline(
                prompt=prompt,
                negative_promp=negative_prompt,
                image=input_img,
                mask_image=mask_img,
                strength=strength,
                guidance_scale=cfg,
                generator=generator,
                num_inference_steps=steps,
            ).images[0]

        else:

            img = self.pipeline(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                image=input_img,
                mask_image=mask_img,
                strength=strength,
                guidance_scale=cfg,
                generator=generator,
                num_inference_steps=steps,
            ).images[0]

        if(blend_img):
            img = blend_image(img,input_img,mask_img,3)
                
        return img,seed


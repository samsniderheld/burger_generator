import itertools

import torch
import torch.utils.checkpoint

from transformers import CLIPTextModel, CLIPTokenizer

from argparse import Namespace

import accelerate

from dream_booth_pipeline.dream_booth_functions import training_function, TrainingConfig



pretrained_model_name_or_path = "SG161222/Realistic_Vision_V1.4" 

instance_prompt = "<food-texture> a 2D texture of food ingredients" 

save_path = "/content/template_pipeline/dream_booth_data/imgs"
caption_path = "/content/template_pipeline/dream_booth_data/captions"

# #prior preservation
# prior_preservation = False #@param {type:"boolean"}
# prior_preservation_class_prompt = "a 2D texture of food ingredients" #@param {type:"string"}

# num_class_images = 12
# sample_batch_size = 2
# prior_loss_weight = 0.5
# prior_preservation_class_folder = "./class_images"
# class_data_root=prior_preservation_class_folder
# class_prompt=prior_preservation_class_prompt

# Load models and create wrapper for stable diffusion


config = TrainingConfig()

accelerate.notebook_launcher(training_function, 
                             args=( config))
# for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
#   if param.grad is not None:
#     del param.grad  # free some memory
#   torch.cuda.empty_cache()
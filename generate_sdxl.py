import os
import random
import time
import cv2
import numpy as np

from arg_parser import parse_sdxl_args
from pipelines.pipelines import (ControlnetSDXLPipeline,InpaintingSDXLPipeline)
from utils import load_img_for_sdxl

# Parse arguments from command line or script input
args = parse_sdxl_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Initialize pipelines based on the flags
if args.pipeline_type == 'inpainting':
    sdxl_pipe  = InpaintingSDXLPipeline(args.sdxl_model)
elif args.pipeline_type == 'controlnet':
    sdxl_pipe  = ControlnetSDXLPipeline(args.sdxl_model)

for i in range(args.num_samples):

    start_time = time.time()

    if args.pipeline_type == 'inpainting':
        
        mask_num = random.randint(1,5)

        path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
        base_img = load_img_for_sdxl(path)

        mask_path = os.path.join(args.template_dir,f"{mask_num}_ingredient_mask.png")
        mask_img = load_img_for_sdxl(mask_path)

        img = sdxl_pipe.generate_img(
            args.prompt, 
            args.negative_prompt,
            base_img,
            mask_img,
            1, 
            args.cfg_scale,
            args.steps
        ).images[0]
        
    elif args.pipeline_type == 'controlnet':

        path = os.path.join(args.template_dir,f"{mask_num}_ingredient.png")
        base_img = load_img_for_sdxl(path)

        img = sdxl_pipe(
            args.prompt,
            args.negative_prompt,
            base_img,
            args.controlnet_str,
            args.cfg_scale,
            args.steps
        ).images[0]

    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)




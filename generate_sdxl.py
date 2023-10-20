


import os
import random
import time
import cv2
import numpy as np
from PIL import Image
from arg_parser import parse_sdxl_args
from pipelines.pipelines import (SDXLPipeline)

# Parse arguments from command line or script input
args = parse_sdxl_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load base images

# Initialize pipelines based on the flags
if args.load_from_file:
    sdxl_pipe  = SDXLPipeline(args.sdxl_model,True)
else:
    sdxl_pipe  = SDXLPipeline(args.sdxl_model)

for i in range(args.num_samples):
    start_time = time.time()

    ingredients = args.ingredients

    # Generate burger image with ingredients
    burger_ingredient_string = "".join([f"{ingredient}, " for ingredient in ingredients])
    burger_prompt = (
        f"image a burger with a {burger_ingredient_string} photorealistic photography, 8k uhd, "
        "full framed, photorealistic photography, dslr, soft lighting, high quality, Fujifilm XT3\n\n"
    )
    print(burger_prompt)

    img = sdxl_pipe.generate_img(burger_prompt, args.burger_steps, args.cfg_scale)

    # Save the final burger image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time:.2f} seconds to execute.")
    ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 
    out_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)




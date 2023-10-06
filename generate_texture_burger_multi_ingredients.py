"""WIP multi ingredient gen file"""
import os
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch

from arg_parser import parse_args
from pipelines.pipelines import ControlNetPipeline,Img2ImgPipeline
from utils import(blend_image, composite_ingredients, 
                  generate_template_and_mask,read_ingredients_from_txt)


args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

control_net_img = Image.open(args.input_texture)
overlay_top = Image.open(os.path.join(args.overlay_dir,"top.png")).convert("RGBA")
overlay_bottom = Image.open(os.path.join(args.overlay_dir,"bottom.png")).convert("RGBA")


if(args.gen_texture):
    controlnet_pipe = ControlNetPipeline(args.base_texture_model, args.controlnet_path)
if(args.gen_burger):
    img2img_pipe = Img2ImgPipeline(args.base_img2img_model)

for i in range(args.num_samples):

    start_time = time.time()

    #set up ingredient list
    ingredient_prompt_embeds = []

    if args.txt_file != None:
        all_ingredients = read_ingredients_from_txt(args.txt_file)
        ingredients = random.choices(all_ingredients,k=random.randint(2,5))

    else:
        ingredients = args.ingredients

    if(args.gen_texture):
        #generated ingredient texture/s
        textures = []
        for ingredient in ingredients:

            prompt = f"""food-texture, a 2D texture of {ingredient}+++, layered++, side view, 
            full framed, photorealistic photography, 8k uhd, dslr, soft lighting, high quality, 
            Fujifilm XT3\n\n"""

            texture = controlnet_pipe.generate_img(prompt,
                            control_net_img,
                            args.controlnet_str,
                            args.dims,
                            args.dims,
                            args.texture_steps,
                            args.cfg_scale)
            
            if(args.gen_burger): 
                textures.append(texture)

            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"The script took {elapsed_time:.2f} seconds to execute.")

                out_img = cv2.cvtColor(np.uint8(texture),cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{args.output_dir}/{ingredient}_{i:04d}.jpg", out_img)

    if(args.gen_burger):

         #create templates, values, and mask
        template,template_values,mask = generate_template_and_mask(len(ingredients), overlay_top, overlay_bottom)

        burger_ingredient_string = "".join([f"{ingredient}, " for ingredient in ingredients]) 

        burger_prompt = f"""image a burger with a burger king beef patty+++, {burger_ingredient_string}, poppyseed bun+++, photorealistic photography, 
        8k uhd, full framed, photorealistic photography, dslr, soft lighting, 
        high quality, Fujifilm XT3\n\n"""

        print(burger_prompt)

        if(args.gen_texture):

            ingredient_textures = textures[::-1]
        
        else:
            directory_path = args.texture_dir

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

            file_urls = sorted(
                [
                    os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                    if os.path.isfile(os.path.join(directory_path, f)) and 
                    os.path.splitext(f)[1].lower() in image_extensions
                ],
                key=str.casefold  # This will sort the URLs in a case-insensitive manner
            )
            all_ingredients = [os.path.basename(file).split("_")[0] for file in file_urls]

            all_ingredients = list(set(all_ingredients))

            num_ingredients = random.randint(1,len(all_ingredients))
            ingredients = random.sample(all_ingredients, num_ingredients)

            ingredient_textures = []

            for ingredient in ingredients:

                texture_paths = [path for path in file_urls if ingredient in  path]

                texture_path = random.choice(texture_paths)

                texture = Image.open(texture_path)

                ingredient_textures.append(texture)
        

        input_img = composite_ingredients(ingredient_textures[::-1],template,template_values)

        img = img2img_pipe.generate_img(burger_prompt,
                        input_img,
                        args.img2img_strength,
                        args.burger_steps,
                        args.cfg_scale)
        
        img = blend_image(img,input_img,mask,args.mask_blur)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        ingredient_string = "".join([f"{ingredient}_" for ingredient in ingredients]) 

        out_img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{args.output_dir}/{ingredient_string}_{i:4d}.jpg", out_img)

        pipeline_img = np.hstack([template,input_img, img.convert('RGB')])
        pipeline_img = cv2.cvtColor(np.uint8(pipeline_img),cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"pipeline_img.jpg", pipeline_img)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

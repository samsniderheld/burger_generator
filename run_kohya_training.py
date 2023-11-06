import argparse
import json
import subprocess
import os
import shutil
import random
import torch
import gc
from PIL import Image, ImageDraw
from pipelines.pipelines import (InpaintingSDXLPipeline)
from utils import read_ingredients_from_txt
from diffusers import StableDiffusionXLPipeline

def json_to_params(path):

    with open(path, 'r') as file:
        params = json.load(file)
    return params["params"]


def run_training_script(params):
    """Run the training script with the specified parameters."""

    conf_text = f"""[sdxl_arguments]
        cache_text_encoder_outputs = false
        no_half_vae = true
        min_timestep = 0
        max_timestep = 1000
        shuffle_caption = false

        [model_arguments]
        pretrained_model_name_or_path = '{params['base_model']}'
        vae = "/content/kohya-trainer/finetune/SDXL/sdxl_vae.safetensors"

        [dataset_arguments]
        debug_dataset = false
        in_json = '{params['json_file']}'
        train_data_dir = '{params['train_data_dir']}'
        dataset_repeats = 1
        keep_tokens = 0
        resolution = "1024,1024"
        caption_dropout_rate = 0
        caption_tag_dropout_rate = 0
        caption_dropout_every_n_epochs = 0
        color_aug = false
        token_warmup_min = 1
        token_warmup_step = 0

        [training_arguments]
        output_dir = "/content/output"
        output_name = '{params['new_model_name']}'
        save_precision = "fp16"
        save_every_n_steps = 10000
        train_batch_size = 4
        max_token_length = 75
        mem_eff_attn = false
        xformers = true
        max_train_steps = {params['max_train_steps']}
        max_data_loader_n_workers = 8
        persistent_data_loader_workers = true
        gradient_checkpointing = true
        gradient_accumulation_steps = 1
        mixed_precision = "fp16"

        [logging_arguments]
        log_with = "tensorboard"
        logging_dir = "/content/logs"
        log_prefix = "sdxl_finetune"

        [sample_prompt_arguments]
        sample_every_n_steps = {params['sample_every_step']}
        sample_sampler = "euler_a"

        [saving_arguments]
        save_model_as = "safetensors"

        [optimizer_arguments]
        optimizer_type = "AdaFactor"
        learning_rate = 4e-7
        train_text_encoder = false
        max_grad_norm = 0
        optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False",]
        lr_scheduler = "constant_with_warmup"
        lr_warmup_steps = 100
        """

    with open("config_file.toml", 'w') as new_file:
        new_file.write(conf_text)

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=1",
        "--config_file", "/content/kohya-trainer/accelerate_config/config.yaml",
        "./sdxl_train.py",
        "--sample_prompts","/content/kohya-trainer/sample_prompt.toml" ,
        "--config_file", "/content/kohya-trainer/config_file.toml"  
    ]

    print("\n\n================")


    print(cmd)
    subprocess.run(cmd)

    print("\n\nTesting Model")

    sample_dir_path = os.path.join(params['output_dir'],params['new_model_name'],"samples")
    model_dir_path = os.path.join(params['output_dir'],params['new_model_name'])
    os.makedirs(sample_dir_path, exist_ok=True)

    sd_path = f"/content/output/{params['new_model_name']}"

    
    safe_tensor_path = f"{sd_path}.safetensors"

    pipe = StableDiffusionXLPipeline.from_single_file(
      safe_tensor_path,
      torch_dtype=torch.float16, variant="fp16",
      use_safetensors=True
    )

    pretrain_path = sd_path

    pipe.save_pretrained(pretrain_path)

    sdxl_pipe  = InpaintingSDXLPipeline(sd_path)

    test_prompts = read_ingredients_from_txt("test_captions.txt")

    for prompt in test_prompts:
        print(prompt)

        negative_prompt = "poor quality, unappetizing"

        mask_num = random.randint(1,5)
        mask_num = 3

        path = f"/content/kohya-trainer/burger_templates/{mask_num}_ingredient.png"
        base_img = Image.open(path)
        base_img = base_img.convert("RGB")
        base_img = base_img.resize((1024,1024))

        mask_path = f"/content/kohya-trainer/burger_templates/{mask_num}_ingredient_mask.png"
        mask_img = Image.open(mask_path)
        mask_img = mask_img.convert("RGB")
        mask_img = mask_img.resize((1024,1024))

        img = sdxl_pipe.generate_img(prompt, 
            negative_prompt,
            base_img,
            mask_img,
            1, 
            7, 
            20
        )

        draw_img = ImageDraw.Draw(img)
        draw_img.text((50,50),prompt, fill=(255,0,0))

        save_path = f"{sample_dir_path}/{prompt.replace('.','').replace(' ','')}_{params['new_model_name']}.jpg"
        img.save(save_path)

    del pipe
    del sdxl_pipe
    gc.collect()
    torch.cuda.empty_cache()


    shutil.copy(safe_tensor_path,model_dir_path)
    shutil.rmtree("/content/output/")

def main(config_path):

    param_combinations = json_to_params(config_path)

    print(param_combinations)

    for params in param_combinations:
        run_training_script(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script with Config File")
    parser.add_argument("--config", default="training_config.json", help="Path to the JSON config file")
    args = parser.parse_args()

    main(args.config)

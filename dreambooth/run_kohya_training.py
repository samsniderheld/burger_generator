import argparse
import json
import subprocess

import shutil
import random

from PIL import Image
from pipelines.pipelines import (InpaintingSDXLPipeline)
from utils import read_ingredients_from_txt

def json_to_params(path):

    with open(path, 'r') as file:
        params = json.load(file)

    return params


def run_training_script(params):
    """Run the training script with the specified parameters."""

    cmd_str = (
        f"--cache_text_encoder_outputs=false"
        f"--no_half_vae=true"
        f"--min_timestep=0"
        f"--max_timestep=1000"
        f"--shuffle_caption=false"
        f"--pretrained_model_name_or_path='{params.base_model}'"
        f"--vae='/content/kohya-trainer/finetune/SDXL/sdxl_vae.safetensors'"
        f"--debug_dataset=false"
        f"--in_json='/content/kohya-trainer/finetune/SDXL/meta_lat.json'"
        f"--train_data_dir='{params.train_data_dir}'"
        f"--dataset_repeats=1"
        f"--keep_tokens=0"
        f"--resolution='1024,1024'"
        f"--caption_dropout_rate=0"
        f"--caption_tag_dropout_rate=0"
        f"--caption_dropout_every_n_epochs=0"
        f"--color_aug=false"
        f"--token_warmup_min=1"
        f"--token_warmup_step=0"
        f"--output_dir='/content/output'"
        f"--output_name='{params.new_model_name}'"
        f"--save_precision='fp16'"
        f"--save_every_n_steps='{params.max_train_steps}'"
        f"--train_batch_size=4"
        f"--max_token_length=75"
        f"--mem_eff_attn=false"
        f"--xformers=true"
        f"--max_train_steps='{params.max_train_steps}'"
        f"--max_data_loader_n_workers=8"
        f"--persistent_data_loader_workers=true"
        f"--gradient_checkpointing=true"
        f"--gradient_accumulation_steps=1"
        f"--mixed_precision='fp16'"
        f"--log_with='tensorboard'"
        f"--logging_dir='/content/logs'"
        f"--log_prefix='sdxl_finetune'"
        f"--sample_every_n_steps='{params.sample_steps}'"
        f"--sample_sampler='euler_a'"
        f"--save_model_as='diffusers'"
        f"--optimizer_type='AdaFactor'"
        f"--learning_rate=4e-7"
        f"--train_text_encoder=false"
        f"--max_grad_norm=0"
        f"--optimizer_args=[ 'scale_parameter=False', 'relative_step=False', 'warmup_init=False',]"
        f"--lr_scheduler='constant_with_warmup'"
        f"--lr_warmup_steps=100"
    )

    cmd_str += " ".join(params)

    print("\n\n================")
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

    print("\n\nTesting Model")

    sdxl_pipe  = InpaintingSDXLPipeline(params.sdxl_model)

    test_prompts = read_ingredients_from_txt("test_captions.txt")

    for prompt in test_prompts:
        print(prompt)

        negative_prompt = "poor quality, unappetizing"

        mask_num = random.randint(1,5)

        path = f"/content/burger_generator/burger_templates/{mask_num}_ingredient.png"
        base_img = Image.open(path)
        base_img = base_img.convert("RGB")
        base_img = base_img.resize((1024,1024))

        mask_path = f"/content/burger_generator/burger_templates/{mask_num}_ingredient_mask.png"
        mask_img = Image.open(mask_path)
        mask_img = mask_img.convert("RGB")
        mask_img = mask_img.resize((1024,1024))

        img = sdxl_pipe.generate_img(prompt, 
            negative_prompt,
            base_img,
            mask_img,
            1, 
            20, 
            5
        )

        save_path = f"{params.output_dir}/{prompt}.jpg"
        img.save(save_path)
        
    shutil.rmtree(params.sdxl_model)

def main(config_path):

    param_combinations = json_to_params(config_path)

    for params in param_combinations:
        run_training_script(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script with Config File")
    parser.add_argument("--config", default="training_config.json",required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    main(args.config)








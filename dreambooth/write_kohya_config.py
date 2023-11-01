import argparse

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A quick script to rewrite a kohya training conf"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--base_model', type=str, default='sd_xl_base_0.9.safetensors', 
        help='The SD model we are using')
    parser.add_argument(
        '--train_data_dir', type=str, default='sdxl_dataset', 
        help='The SD model we are using')
    parser.add_argument(
        '--new_model_name', type=str, default='kohya_fine_tune', 
        help='The SD model we are using')
    parser.add_argument(
        '--max_train_steps', type=int, default=2500,
        help='The min num of ingredients')
    parser.add_argument(
        '--sample_steps', type=int, default=100,
        help='The max num of ingredients')

    return parser.parse_args()


args = parse_args()

conf_text = f"""[sdxl_arguments]
cache_text_encoder_outputs = false
no_half_vae = true
min_timestep = 0
max_timestep = 1000
shuffle_caption = false

[model_arguments]
pretrained_model_name_or_path = "{args.base_model}"
vae = "/content/kohya-trainer/finetune/SDXL/sdxl_vae.safetensors"

[dataset_arguments]
debug_dataset = false
in_json = "/content/kohya-trainer/finetune/SDXL/meta_lat.json"
train_data_dir = "{args.train_data_dir}"
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
output_name = "{args.new_model_name}"
save_precision = "fp16"
save_every_n_steps = "{args.max_train_steps}"
train_batch_size = 4
max_token_length = 75
mem_eff_attn = false
xformers = true
max_train_steps = {args.max_train_steps}
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
sample_every_n_steps = {args.sample_steps}
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
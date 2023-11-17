import argparse
import json



parser = argparse.ArgumentParser(description="Dynamically Create Json File")
parser.add_argument(
        '--base_model', type=str, default='/content/kohya-trainer/finetune/SDXL/sd_xl_base_0.9.safetensors', 
        help='The base model to start with.')
parser.add_argument(
        '--train_data_dir', type=str, default='/content/kohya-trainer/finetune/SDXL/sd_xl_base_0.9.safetensors', 
        help='The folder containing our dataset.')
parser.add_argument(
        '--output_dir', type=str, default='/content/gdrive/MyDrive/Section_11/Burger_King/experiments', 
        help='where the outputs go')
parser.add_argument(
        '--dataset_sizes', nargs="+", type=int, default=[5,10,23],
        help='The our dataset.')
parser.add_argument(
        '--step_sizes', nargs="+", type=int, default=[1000,2000,2500],
        help='The our dataset.')
parser.add_argument(
        '--sample_every_step', type=int, default=1000,
        help='The our dataset.')
parser.add_argument(
        '--output_json_path', type=str, default='training_config.json', 
        help='where the outputs go')

args = parser.parse_args()

all_configs = []

for dataset_size in args.dataset_sizes:

    for step_size in args.step_sizes:

        base_config = {
                    "base_model": args.base_model,
                    "train_data_dir": f"/content/gdrive/MyDrive/Section_11/Burger_King/experiments/bk_{dataset_size}_samples_{step_size}_steps/dataset",
                    "output_dir": args.output_dir,
                    "json_file": f"/content/gdrive/MyDrive/Section_11/Burger_King/experiments/bk_{dataset_size}_samples_{step_size}_steps/meta_lat_{dataset_size}_samples.json",
                    "new_model_name": f"bk_{dataset_size}_samples_{step_size}_steps",
                    "max_train_steps": step_size,
                    "sample_every_step": 10000,
                    "num_samples": dataset_size
                }
        
        all_configs.append(base_config)

output_json = {
    "params" : all_configs
}

# Serializing json
json_object = json.dumps(output_json, indent=4)
 
# Writing to sample.json
with open(args.output_json_path, "w") as outfile:
    outfile.write(json_object)
import argparse

def parse_args():
    """
    Parses the command-line arguments for the script.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating burgers with multiple ingredients"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--sdxl_model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', 
        help='The SD model we are using')
    parser.add_argument(
        '--food_list', type=str, default="assets/food_list.txt", 
        help='the food list we are using to generate')
    parser.add_argument(
        '--num_ingredients', type=int, default=3, 
        help='number of random ingredients')
    parser.add_argument(
        '--template_dir', 
         type=str, default='burger_templates', 
        help='Which template dir')
    parser.add_argument(
        '--prompt', 
         type=str, 
        help='the prompt')
    parser.add_argument(
        '--negative_prompt', 
         type=str, 
        help='the negative prompt')
    parser.add_argument(
        '--steps', type=int, default=20, 
        help='The number of diffusion steps for burger generation')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--cfg_scale', type=float, default=4.5, 
        help='How much creativity the pipeline has')
    return parser.parse_args()

def parse_pregen_args():
    """
    Parses the command-line arguments for the script.
    """
    desc = "parameter for generating the pregen list"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--white_list_path', type=str, default='assets/food_list.txt', 
        help='the path of the white list')
    parser.add_argument(
        '--dataset_path', type=str, default="assets/ingredients.pkl", 
        help='the ingredient dataset')
    parser.add_argument(
        '--embedding_model_path', type=str, default="all-mpnet-base-v2", 
        help='the text encoder')
    parser.add_argument(
        '--ingredient_frequecy_cuttoff', type=int, default=50, 
        help='calculate probability of ingredients that show up more than this val')
    parser.add_argument(
        '--num_recipes', type=int, default=100000, 
        help='how many recipes are we generating')
    parser.add_argument(
        '--recipe_frequecy_cuttoff', type=int, default=10, 
        help='only save recipes that show up more often')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    
    return parser.parse_args()
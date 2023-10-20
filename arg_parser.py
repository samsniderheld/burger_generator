import argparse

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating burgers with multiple ingredients"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--input_texture', type=str, default='input_templates/00.jpg', 
        help='The directory for input data')
    parser.add_argument(
        '--overlay_dir', type=str, default='burger_templates/', 
        help='The burger overlay')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--texture_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_texture_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--base_img2img_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using for img2img')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-scribble', 
        help='The controlnet model we are using.')
    parser.add_argument(
        '--ingredients', type=str, nargs='+', default=['fried chicken', 'raspberries'],
        help='The ingredients we are generating')
    parser.add_argument(
        '--min_num_ingredients', type=int, default=1,
        help='The min num of ingredients')
    parser.add_argument(
        '--max_num_ingredients', type=int, default=4,
        help='The max num of ingredients')
    parser.add_argument(
        '--texture_steps', type=int, default=20, 
        help='The number of diffusion steps for texture generation')
    parser.add_argument(
        '--burger_steps', type=int, default=20, 
        help='The number of diffusion steps for burger generation')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--controlnet_str', type=float, default=.85, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--img2img_strength', type=float, default=.2, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--mask_blur', type=int, default=3, 
        help='How to blur mask composition')
    parser.add_argument(
        '--cfg_scale', type=float, default=3.5, 
        help='How much creativity the pipeline has')
    parser.add_argument(
        '--txt_file', type=str, default=None, 
        help='The ingredient texture swe want to generate.')
    parser.add_argument(
        '--gen_texture', action='store_true',
        help='generate textures')
    parser.add_argument(
        '--gen_burger', action='store_true',
        help='generate burgers')
    parser.add_argument(
        '--use_SDXL', action='store_true',
        help='use SDXL')

    return parser.parse_args()

def parse_gradio_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Hugging Face Diffusers Pipeline for generating ingredient textures"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--input_texture', type=str, default='input_templates/00.jpg', 
        help='The directory for input data')
    parser.add_argument(
        '--template', type=str, default='burger_templates/burger_template.png', 
        help='The burger template')
    parser.add_argument(
        '--overlay_dir', type=str, default='burger_templates/', 
        help='The burger overlay')
    parser.add_argument(
        '--mask', type=str, default='burger_templates/burger_mask.png', 
        help='The burger mask')
    parser.add_argument(
        '--output_dir', type=str, default='burger_outputs', 
        help='The directory for all the output results.')
    parser.add_argument(
        '--base_texture_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using')
    parser.add_argument(
        '--base_img2img_model', type=str, default='SG161222/Realistic_Vision_V1.4', 
        help='The SD model we are using for img2img')
    parser.add_argument(
        '--controlnet_path', type=str, default='lllyasviel/sd-controlnet-scribble', 
        help='The controlnet model we are using.')
    parser.add_argument(
        '--ingredient', type=str, default='avocado', 
        help='The ingredient texture we want to generate.')
    parser.add_argument(
        '--steps', type=int, default=100, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--dims', type=int, default=512, 
        help='The Dimensions to Render at')
    parser.add_argument(
        '--controlnet_str', type=float, default=.85, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--img2img_strength', type=float, default=.4, 
        help='How much impact does the control net have.')
    parser.add_argument(
        '--mask_blur', type=int, default=3, 
        help='How to blur mask composition')
    parser.add_argument(
        '--cfg_scale', type=float, default=4.5, 
        help='How much creativity the pipeline has')
    parser.add_argument(
        '--use_SDXL', action='store_true',
        help='use SDXL')

    return parser.parse_args()

def parse_sdxl_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
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
        '--ingredients', type=str, nargs='+', default=['fried chicken', 'raspberries'],
        help='The ingredients we are generating')
    parser.add_argument(
        '--burger_steps', type=int, default=20, 
        help='The number of diffusion steps for burger generation')
    parser.add_argument(
        '--num_samples', type=int, default=1, 
        help='The number of diffusion steps')
    parser.add_argument(
        '--cfg_scale', type=float, default=4.5, 
        help='How much creativity the pipeline has')

    return parser.parse_args()
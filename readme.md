# README Documentation for Generate_SDXL.py

## Overview
`Generate_SDXL.py` is a Python script designed for bulk generation of burger images. It combines various ingredients, both standard and random, to create a diverse set of burger samples. The script is capable of enforcing a specific ratio of standard to random ingredients and can incorporate special templates for specific ingredients like an extra patty.

## Features
- **Variable Sample Generation:** Generates a user-defined number of burger samples.
- **Ingredient Customization:** Supports a mix of standard and random ingredients, with an enforced 2:1 ratio.
- **Special Templates:** Uses a unique template for burgers with an extra patty.
- **Storage and Tracking:** Saves generated burger images in a specified directory and records their parameters in a JSON file.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PIL (Python Imaging Library)
- Other dependencies as required by `InpaintingSDXLPipeline` and utility modules.

## Usage
1. **Argument Parsing:** The script starts by parsing arguments provided via command line or script input. These arguments include the number of samples, the directory for output, and paths to various resources.

2. **Ingredient Selection:**
   - Random ingredients are read from a provided text file.
   - A predefined list of standard ingredients is available in the script.

3. **Directory Setup:** The output directory is created if it does not exist.

4. **Image Generation Loop:** For each sample:
   - A random number of ingredients is selected.
   - Prompts are constructed based on ingredient selection.
   - An image and a mask are loaded for inpainting.
   - The SDXL pipeline generates the burger image based on the prompts.
   - Image details are stored, and the image is labeled and saved.

5. **Saving Sample Details:** All sample details are saved in a JSON file in the output directory.

## Example Command
To run the script, use a command in the following format (assuming all required arguments are defined in `arg_parser`):
```
python Generate_SDXL.py --num_samples 50 --output_dir ./output --food_list ./ingredients.txt
```

## Output
- Generated burger images will be saved in the specified output directory.
- A JSON file (`samples.json`) containing details of all samples will also be saved in this directory.

## Note
Ensure all dependencies are installed and paths to templates, ingredient lists, and other resources are correctly set in the arguments.
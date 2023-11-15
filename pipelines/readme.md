# README Documentation for pipelines.py

## Overview
`pipelines.py` is a Python module that provides utility classes and functions for image generation using various pipelines. These pipelines leverage pre-trained models from the `diffusers` and `compel` modules to produce images based on provided textual prompts. The module simplifies the process of iterating through image generation loops.

## Features
- **Support for Multiple Pipelines:** Interfaces with models from `diffusers` and `compel` for image generation.
- **InpaintingSDXLPipeline Class:** Facilitates image inpainting with specific configurations and enhancements.
- **Flexible Prompt Handling:** Supports chunking of prompts and negative prompts for improved image generation.
- **Seed-based Generation:** Allows for reproducible image generation using random seeds.
- **Image Blending Option:** Provides functionality to blend generated images with input images.

## Dependencies
- Python 3.x
- PyTorch (`torch`)
- `diffusers` module
- `compel` module
- Additional utility modules (`basic_utils`, `burger_gen_utils`)

## InpaintingSDXLPipeline Class
### Initialization
- **Parameters:**
  - `pipeline_path`: Path to the pre-trained pipeline model.
  - `use_freeU` (optional): Flag to enable FreeU configuration.
- **Functionality:**
  - Loads and configures the inpainting pipeline upon initialization.

### Methods
- **load_pipeline:**
  - Loads the inpainting pipeline and sets up the Compel module.
  - Configures the pipeline with specific scheduler settings.
- **generate_img:**
  - **Parameters:**
    - `prompt`: Textual prompt for image generation.
    - `negative_prompt`: Negative prompt for guiding the generation.
    - `input_img`: Base image for inpainting.
    - `mask_img`: Mask image for inpainting.
    - `strength`: Inpainting strength parameter.
    - `cfg`: Guidance scale for the generation.
    - `steps`: Number of inference steps.
    - `use_chunking` (optional): Flag to enable prompt chunking.
    - `blend_img` (optional): Flag to blend the generated image with the input image.
  - **Functionality:**
    - Generates an image based on the provided prompts and inpainting parameters.
    - Returns the generated image and the used random seed.

## Usage
- Import `InpaintingSDXLPipeline` from `pipelines`.
- Initialize the pipeline with the required model path.
- Generate images by passing appropriate prompts, images, masks, and other parameters to the `generate_img` method.

## Example
```python
from pipelines import InpaintingSDXLPipeline

# Initialize the pipeline
pipeline = InpaintingSDXLPipeline("path_to_pretrained_model")

# Generate an image
generated_image, seed = pipeline.generate_img(
    prompt="A burger with lettuce and tomatoes",
    negative_prompt="No fish or chicken",
    input_img=input_image,
    mask_img=mask_image,
    strength=0.8,
    cfg=7.5,
    steps=50
)
```

## Note
Ensure all required dependencies are installed. The `pipeline_path` should point to a valid model compatible with the `diffusers` module.
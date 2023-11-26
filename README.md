# ComfyUI-safety-checker
A NSFW/Safety Checker Node for ComfyUI

## Safety Checker
Using the model from https://huggingface.co/CompVis/stable-diffusion-safety-checker

The node checks images for nsfw concepts if it detects an nsfw concept it produces a black image and boolean with concept scores.
Just plug it between your vae decode and saveimage/previewimage nodes.

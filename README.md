# ComfyUI-safety-checker
A NSFW/Safety Checker Node for ComfyUI

## Safety Checker
Using the model from https://huggingface.co/CompVis/stable-diffusion-safety-checker

The node checks images for nsfw concepts if it detects an nsfw concept it produces a black image and boolean with concept scores.
Just plug it between your vae decode and saveimage/previewimage nodes.

### Node Setup
![image](https://github.com/42lux/ComfyUI-safety-checker/assets/7535793/d9b344d2-5c24-48ae-8728-53ab8b396190)

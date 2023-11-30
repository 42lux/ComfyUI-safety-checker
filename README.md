# ComfyUI-safety-checker
A NSFW/Safety Checker Node for ComfyUI.

This node introduces a CLIP-based safety checker for identifying and handling Not Safe For Work (NSFW) content in images. It utilizes a pretrained model from Hugging Face https://huggingface.co/CompVis/stable-diffusion-safety-checker, specifically designed for safety checks in images. The primary objective is to ensure that generated or processed images adhere to safety guidelines by detecting and replacing inappropriate content.

## Safety Checker Node
This node processes input images, either by directly connecting an image or via VAE Decode, to identify any NSFW (Not Safe For Work) content. Detected NSFW images are substituted with solid black placeholders.

### Sensitivity Adjustment
The sensitivity level of the NSFW detection can be modified. Standard Value is 0.5.

| Sensitivity Level | Description                                       |
|-------------------|---------------------------------------------------|
| 0                 | No filtering                                      |
| **0.5**           | **Detection threshold for explicit nudity**       |
| 1.0               | Sensitivity for detecting images with lingerie/underwear |

### Example Node Setup
![image](https://github.com/42lux/ComfyUI-safety-checker/assets/7535793/6847b6e1-63f9-4533-8878-34d481f6bf54)


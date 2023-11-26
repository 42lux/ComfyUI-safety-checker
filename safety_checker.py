import numpy as np
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import comfy.utils
from folder_paths import models_dir
import os

class Safety_Checker:
    def __init__(self):
        self.safety_feature_extractor = None
        self.safety_checker = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "nsfw")
    FUNCTION = "nsfw_checker"
    CATEGORY = "image"

    def numpy_to_pil(self, images):
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def nsfw_checker(self, images):
        safety_checker_model = os.path.join(models_dir, "safety_checker")
        if self.safety_feature_extractor is None:
            self.safety_feature_extractor = CLIPImageProcessor.from_pretrained(safety_checker_model)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker_model)
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(images), return_tensors="pt")
        checked_image, nsfw = self.safety_checker(images=images, clip_input=safety_checker_input.pixel_values)
        return checked_image, nsfw

    def censor_batch(self, x):
        x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
        x_checked_image, has_nsfw_concept = self.nsfw_checker(x_samples_ddim_numpy)
        x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        return x

NODE_CLASS_MAPPINGS = {
    "Safety Checker": Safety_Checker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
}

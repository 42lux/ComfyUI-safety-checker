import os
import logging
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPConfig, CLIPVisionModel, PreTrainedModel
import comfy.utils
from folder_paths import models_dir

# Define ANSI escape codes for coloring logs
RED = "\033[91m"
RESET = "\033[0m"

# Custom logging formatter for adding colors to warnings and above
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"{RED}{super().format(record)}{RESET}"
        return super().format(record)

# Configure logging with custom formatter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# Define a CLIP-based safety checker. Used model https://huggingface.co/CompVis/stable-diffusion-safety-checker
class ClipSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        # Predefined embeddings and weights for concepts and special care
        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False) 
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @staticmethod
    def compute_cosine_similarity(embeds, target_embeds):
        # Compute the cosine similarity between two sets of embeddings
        # The embeddings are expanded along a new dimension for proper broadcasting
        return F.cosine_similarity(embeds.unsqueeze(1), target_embeds, dim=2)

    @staticmethod
    def process_image_scores(image_idx, special_scores, concept_scores):
        # Process and compile scores for a single image
        # Initialize dictionaries to store scores and identified concepts
        result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

        # Identify indices with positive special and concept scores
        special_attention_indices = torch.where(special_scores > 0)[0]
        concept_attention_indices = torch.where(concept_scores > 0)[0]

        # Store and round special scores
        for idx in special_attention_indices:
            score = round(special_scores[idx].item(), 3)
            result_img["special_scores"][idx.item()] = score
            result_img["special_care"].append((idx.item(), score))

        # Store and round concept scores
        for idx in concept_attention_indices:
            score = round(concept_scores[idx].item(), 3)
            result_img["concept_scores"][idx.item()] = score
            result_img["bad_concepts"].append(idx.item())

        return result_img

    @staticmethod
    def replace_nsfw_images(images, nsfw_concepts):
        # Replace NSFW images in a batch with blank images
        for idx, is_nsfw in enumerate(nsfw_concepts):
            if is_nsfw:
                # Replace the image with a black image of the same size
                images[idx] = torch.zeros_like(images[idx]) if torch.is_tensor(images) or torch.is_tensor(images[0]) else np.zeros(images[idx].shape)
        return images

    @staticmethod
    def process_image_scores_batch(special_scores, concept_scores):
        # Process and compile scores for a batch of images
        batch_results = []
        for i in range(special_scores.shape[0]): 
            # Process each image in the batch
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
            special_attention_indices = torch.where(special_scores[i] > 0)[0]
            concept_attention_indices = torch.where(concept_scores[i] > 0)[0]

            # Extract and round scores for each concept and special category
            for idx in special_attention_indices:
                score = round(special_scores[i][idx].item(), 3)
                result_img["special_scores"][idx.item()] = score
                result_img["special_care"].append((idx.item(), score))

            for idx in concept_attention_indices:
                score = round(concept_scores[i][idx].item(), 3)
                result_img["concept_scores"][idx.item()] = score
                result_img["bad_concepts"].append((idx.item(), score))

            batch_results.append(result_img)

        return batch_results

    def forward(self, clip_input, images, sensitivity):
        try:
            with torch.no_grad():
                # Perform forward pass of the vision model to get image embeddings
                image_batch = self.vision_model(clip_input)[1]
                image_embeds = self.visual_projection(image_batch)

                # Compute cosine distances for special care and concept embeddings
                special_cos_dist = ClipSafetyChecker.compute_cosine_similarity(image_embeds, self.special_care_embeds)
                cos_dist = ClipSafetyChecker.compute_cosine_similarity(image_embeds, self.concept_embeds)

                # Adjust scores based on input sensitivity
                adjusted_sensitivity = calculate_sensitivity(sensitivity)
                special_scores = special_cos_dist - self.special_care_embeds_weights.unsqueeze(0) + adjusted_sensitivity
                concept_scores = cos_dist - self.concept_embeds_weights.unsqueeze(0) + adjusted_sensitivity

                # Process scores for the entire batch
                results = ClipSafetyChecker.process_image_scores_batch(special_scores, concept_scores)

                # Check for NSFW content and replace corresponding images
                nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in results]
                images = ClipSafetyChecker.replace_nsfw_images(images, nsfw_concepts)

                # Log a warning if any NSFW content is detected
                if any(nsfw_concepts):
                    logging.warning("Safety Checker: NSFW content detected. Replaced with black output.")

                return images, nsfw_concepts

        except Exception as e:
            logger.error(f"Error in ClipSafetyChecker forward method: {e}")
            raise

# Function to calculate adjusted sensitivity based on input
def calculate_sensitivity(input_sensitivity):
    return -0.1 + 0.14 * input_sensitivity

class Safety_Checker:
    def __init__(self):
        try:
            safety_checker_model = os.path.join(models_dir, "safety_checker")
            self.safety_feature_extractor = CLIPImageProcessor.from_pretrained(safety_checker_model)
            self.safety_checker = ClipSafetyChecker.from_pretrained(safety_checker_model)
        except Exception as e:
            logger.error(f"Error initializing Safety_Checker: {e}")
            raise

    # Define input types for the safety checker node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.10}),
            },
        }

    # Define return types and names for the safety checker node
    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "nsfw")
    FUNCTION = "nsfw_checker"
    CATEGORY = "image"

    # Convert numpy array to PIL image
    def numpy_to_pil(self, images):
        try:
            if torch.is_tensor(images):
                images = images.cpu().numpy()
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images
        except Exception as e:
            logger.error(f"Error in numpy_to_pil: {e}")
            raise

    # Check images for NSFW content and process accordingly
    def nsfw_checker(self, images, sensitivity):
        try:
            safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(images), return_tensors="pt")
            checked_image, nsfw = self.safety_checker(images=images, clip_input=safety_checker_input.pixel_values, sensitivity=sensitivity)
            return checked_image, nsfw
        except Exception as e:
            logger.error(f"Error in nsfw_checker: {e}")
            raise

    # Process a batch of images for NSFW content
    def censor_batch(self, x):
        try:
            x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
            x_checked_image, has_nsfw_concept = self.nsfw_checker(x_samples_ddim_numpy)
            x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
            return x
        except Exception as e:
            logger.error(f"Error in censor_batch: {e}")
            raise

# Mapping for node class and display names
NODE_CLASS_MAPPINGS = {
    "Safety Checker": Safety_Checker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
}

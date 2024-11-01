import torch
import numpy as np

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from .clip_utils import DirectionalSimilarity


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load CLIP model
clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

# TODO load images and prompts
src_images = []
tgt_images = []

src_prompts = []
tgt_prompts = []

# calculate directional CLIP similarity
dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder, device)
scores = []

for i in range(len(src_images)):
    original_image = src_images[i]
    original_caption = src_prompts[i]
    edited_image = tgt_images[i]
    modified_caption = tgt_prompts[i]

    similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
    scores.append(float(similarity_score.detach().cpu()))

print(f"CLIP directional similarity: {np.mean(scores)}")
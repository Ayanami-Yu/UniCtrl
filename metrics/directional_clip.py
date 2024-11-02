import torch
import yaml
import numpy as np

from torchvision.io import read_image
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from metrics.clip_utils import DirectionalSimilarity


# specify the model to test
# available models: sd, masactrl, p2p
# available modes: add, rm
config_file = 'metrics/images.yaml'
model = 'sd'
mode = 'rm'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load CLIP model
clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

# load images and prompts
with open(config_file) as f:
    dataset = yaml.safe_load(f)

src_images = [read_image(data[model]['src_image']) for data in dataset[mode].values()]
tgt_images = [read_image(data[model]['tgt_image']) for data in dataset[mode].values()]
src_prompts = [data['src_prompt'] for data in dataset[mode].values()]
tgt_prompts = [data['tgt_prompt'] for data in dataset[mode].values()]

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

# Add: SD = 0.2147, MasaCtrl = 0.0785, P2P = 0.162
# Remove: SD = 0.1414
print(f"CLIP directional similarity: {round(np.mean(scores), 4)}")
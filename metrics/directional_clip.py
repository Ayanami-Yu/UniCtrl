import os

import numpy as np
import torch
import yaml
from torchvision.io import read_image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from metrics.clip_utils import DirectionalSimilarity

# specify the model to test
# available config files: metrics/images.yaml, metrics/videos.yaml
# available modalities: image, video
# available image models: sd, masactrl, p2p, sega, ledits_pp, mdp, cg
# available video models: animatediff, fatezero, tokenflow, vidtome, flatten, v2v_zero
# available modes: add, rm
modality = "video"
model = "v2v_zero"
mode = "rm"
config_file = "metrics/images.yaml" if modality == "image" else "metrics/videos.yaml"

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

if modality == "image":
    label = "sega" if model == "sega" else "default"
    src_images = [
        read_image(data["src_image"][label]) for data in dataset[mode].values()
    ]
    tgt_images = [
        read_image(data["tgt_image"][model]) for data in dataset[mode].values()
    ]
    src_prompts = [data["src_prompt"] for data in dataset[mode].values()]
    if mode == "add":
        tgt_prompts = [data["tgt_prompt"] for data in dataset[mode].values()]
    else:
        tgt_prompts = [data["tgt_prompt"]["default"] for data in dataset[mode].values()]
elif modality == "video":
    src_images = []
    tgt_images = []
    src_prompts = []
    tgt_prompts = []
    for data in dataset[mode].values():
        src_path = data["src_images"]
        tgt_path = data["tgt_images"][model]

        src_images.extend(
            [read_image(os.path.join(src_path, img)) for img in sorted(os.listdir(src_path))]
        )
        tgt_images.extend(
            [read_image(os.path.join(tgt_path, img)) for img in sorted(os.listdir(tgt_path))]
        )
        src_prompts.extend([data["src_prompt"]] * len(os.listdir(src_path)))
        if mode == "add":
            tgt_prompts.extend([data["tgt_prompt"]] * len(os.listdir(tgt_path)))
        else:
            tgt_prompts.extend([data["tgt_prompt"]["default"]] * len(os.listdir(tgt_path)))
else:
    raise ValueError("Unrecognized modality")

# calculate directional CLIP similarity
dir_similarity = DirectionalSimilarity(
    tokenizer, text_encoder, image_processor, image_encoder, device
)
scores = []

for i in range(len(src_images)):
    original_image = src_images[i]
    original_caption = src_prompts[i]
    edited_image = tgt_images[i]
    modified_caption = tgt_prompts[i]

    similarity_score = dir_similarity(
        original_image, edited_image, original_caption, modified_caption
    )
    scores.append(float(similarity_score.detach().cpu()))

# Add (image): SD = 0.2388, MasaCtrl = 0.0956, P2P = 0.1081, SEGA = 0.1369, LEDITS++ = 0.2109, MDP = 0.1836, CG = 0.2255
# Add (video): AnimateDiff = 0.2297, FateZero = 0.0662, TokenFlow = 0.0987, VidToMe = 0.089, FLATTEN = 0.104, Vid2Vid-Zero = 0.1008

# Remove (image): SD = 0.1243, MasaCtrl = 0.0339, P2P = 0.072, SEGA = 0.0944, LEDITS++ = 0.1641, MDP = 0.1205, CG = 0.1274
# Remove (video): AnimateDiff = 0.1735, FateZero = 0.0247, TokenFlow = 0.1184, VidToMe = 0.1223, FLATTEN = -0.0137, Vid2Vid-Zero = 0.1085
print(f"CLIP directional similarity: {round(np.mean(scores), 4)}")

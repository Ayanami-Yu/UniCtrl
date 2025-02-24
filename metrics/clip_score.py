import os

import torch
import yaml
from torchvision.io import read_image

from metrics.clip_utils import calculate_clip_score

# specify the model to test
# available config files: metrics/images.yaml, metrics/videos.yaml
# available modalities: image, video
# available image models: sd, masactrl, p2p, sega, ledits_pp, mdp, cg
# available video models: animatediff, fatezero, tokenflow, vidtome, flatten, v2v_zero, rav
# available modes: add, rm
modality = "video"
model = "rav"
mode = "rm"
config_file = "metrics/images.yaml" if modality == "image" else "metrics/videos.yaml"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load images and prompts
with open(config_file) as f:
    dataset = yaml.safe_load(f)

if modality == "image":
    images = [read_image(data["tgt_image"][model]) for data in dataset[mode].values()]
    if mode == "add":
        prompts = [data["tgt_prompt"] for data in dataset[mode].values()]
    else:
        # NOTE CLIPinv measures similarity between removed concepts and edited images
        prompts = [data["tgt_prompt"]["sd"] for data in dataset[mode].values()]
elif modality == "video":
    # NOTE We haven't measured temporal consistency as it's not strongly
    # related to the scope of our research.
    images = []
    prompts = []
    for data in dataset[mode].values():
        path = data["tgt_images"][model]
        images.extend(
            [read_image(os.path.join(path, img)) for img in sorted(os.listdir(path))]
        )
        if mode == "add":
            prompts.extend([data["tgt_prompt"]] * len(os.listdir(path)))
        else:
            prompts.extend([data["tgt_prompt"]["animatediff"]] * len(os.listdir(path)))
else:
    raise ValueError("Unrecognized modality")

# CLIPsim (image): 
# SD = 35.4878, MasaCtrl = 31.3468, P2P = 30.2197, SEGA = 32.847, 
# LEDITS++ = 33.8337 (hugging face space), 32.1255 (threshold = 0.95, scale = 7), 29.943 (threshold = 0.75, scale = 10), 
# MDP = 32.396, CG = 34.1302

# CLIPsim (video): 
# AnimateDiff = 36.3802, FateZero = 33.8866, TokenFlow = 34.3846, 
# VidToMe = 34.6253, FLATTEN = 34.3304, Vid2Vid-Zero = 34.7291, Rerender-A-Video = 34.2491

# CLIPinv (image): 
# SD = 19.784, MasaCtrl = 24.0757, P2P = 23.3297, SEGA = 21.7271, 
# LEDITS++ = 20.9888 (hugging face space), 23.1463 (threshold = 0.95, scale = 7), 20.1063 (threshold = 0.75, scale = 10), 
# MDP = 21.071, CG = 19.54

# CLIPinv (video): 
# AnimateDiff = 22.9229, FateZero = 26.743, TokenFlow = 22.5713, 
# VidToMe = 23.832, FLATTEN = 25.3677, Vid2Vid-Zero = 21.7432, Rerender-A-Video = 24.2228
clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")

import os

import torch
import yaml
from torchvision.io import read_image

from metrics.clip_utils import calculate_clip_score

# specify the model to test
# available config files: metrics/images.yaml, metrics/videos.yaml
# available modalities: image, video
# available models: sd, masactrl, p2p, animatediff, fatezero
# available modes: add, rm
modality = "video"
model = "animatediff"
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
        label = "sd" if model == "sd" else "default"
        prompts = [data["tgt_prompt"][label] for data in dataset[mode].values()]
elif modality == "video":
    # NOTE We haven't measured temporal consistency as it's not strongly
    # related to the scope of our research.
    images = []
    prompts = []
    for data in dataset[mode].values():
        path = data["tgt_images"][model]
        images.extend([read_image(os.path.join(path, img)) for img in os.listdir(path)])
        if mode == "add":
            prompts.extend([data["tgt_prompt"]] * len(os.listdir(path)))
        else:
            label = "animatediff" if model == "animatediff" else "default"
            prompts.extend([data["tgt_prompt"][label]] * len(os.listdir(path)))
else:
    raise ValueError("Unrecognized modality")

# Without inversion:
# Add (image): SD = 35.3821, MasaCtrl = 31.809, P2P = 34.6683
# Add (video): AnimateDiff = 36.3802, FateZero = 33.8866
# Remove: SD = 28.6935

# With inversion:
# Add (image): SD = 35.4878, MasaCtrl = 31.3468, P2P = 30.2197
# Add (video): AnimateDiff = 36.3802, FateZero = 33.8866
# TODO CLIP score for removal is not reasonable
# Remove: SD = 29.495, MasaCtrl = 31.4307, P2P = 32.6418
clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")


# TODO rm new metrics
# sd = 19.784, masactrl = 24.0757, p2p = 23.3297
# animatediff = , fatezero = 26.743
import torch
import yaml
import os
from torchvision.io import read_image
from metrics.clip_utils import calculate_clip_score


# specify the model to test
# available config files: metrics/images.yaml, metrics/videos.yaml
# available modalities: image, video
# available models: sd, masactrl, p2p, animatediff, fatezero
# available modes: add, rm
config_file = "metrics/videos.yaml"
modality = "video"
model = "fatezero"
mode = "add"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load images and prompts
with open(config_file) as f:
    dataset = yaml.safe_load(f)

if modality == "image":
    # TODO refactor
    images = [read_image(data[model]["tgt_image"]) for data in dataset[mode].values()]
    prompts = [data["tgt_prompt"] for data in dataset[mode].values()]
elif modality == "video":
    # TODO refactor according to new videos.yaml
    # NOTE We haven't measured temporal consistency as it's not strongly
    # related to the scope of our research.
    images = []
    prompts = []
    for data in dataset[mode].values():
        path = data[model]["tgt_images"]
        images.extend([read_image(os.path.join(path, img)) for img in os.listdir(path)])
        prompts.extend([data["tgt_prompt"]] * len(os.listdir(path)))
else:
    raise ValueError("Unrecognized modality")

# Add (image): SD = 35.3821, MasaCtrl = 31.809, P2P = 34.6683
# Add (video): AnimateDiff = 36.3802, FateZero = 33.8866
# Remove: SD = 28.6935
clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")

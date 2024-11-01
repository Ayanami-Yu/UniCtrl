import torch
import yaml
from torchvision.io import read_image
from metrics.clip_utils import calculate_clip_score


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load images and prompts
with open("metrics/images.yaml") as f:
    dataset = yaml.safe_load(f)

images = [read_image(img) for img in dataset['tgt_images']]
prompts = dataset['tgt_prompts']

clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")  # SD: 14.739

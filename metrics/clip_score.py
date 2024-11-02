import torch
import yaml
from torchvision.io import read_image
from metrics.clip_utils import calculate_clip_score


# specify the model to test
# available models: sd, masactrl, p2p
# available modes: add, rm
config_file = 'metrics/images.yaml'
model = 'sd'
mode = 'rm'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load images and prompts
with open(config_file) as f:
    dataset = yaml.safe_load(f)

images = [read_image(data[model]['tgt_image']) for data in dataset[mode].values()]
prompts = [data['tgt_prompt'] for data in dataset[mode].values()]

# Add: SD = 35.3821, MasaCtrl = 31.809, P2P = 34.6683
# Remove: SD = 28.6935
clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")
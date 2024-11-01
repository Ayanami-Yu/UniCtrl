import torch
import yaml
from torchvision.io import read_image
from metrics.clip_utils import calculate_clip_score


# specify the model to test
config_file = 'metrics/images.yaml'
model = 'masactrl'
mode = 'add'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load images and prompts
with open(config_file) as f:
    dataset = yaml.safe_load(f)

images = [read_image(data[model]['tgt_image']) for data in dataset[mode].values()]
prompts = [data['tgt_prompt'] for data in dataset[mode].values()]

clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")  # SD: 35.3821

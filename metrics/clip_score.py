import torch
from PIL import Image
from .clip_utils import calculate_clip_score


# TODO load images and prompts
images = []
prompts = []

clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {clip_score}")

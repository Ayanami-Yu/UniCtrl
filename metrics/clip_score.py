import torch
from pytorch_lightning import seed_everything
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    """
    Params:
        images: Tensor of shape (N, C, H, W) or a list of tensors of shape (C, H, W)
    """
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    return round(float(clip_score), 4)


# set seed
# TODO start code
seed = 42
seed_everything(seed)

# TODO load images
images = []


prompts = [
    "a corgi",
    "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
]

clip_score = clip_score_fn(images, prompts)

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, _ = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, _ = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


@torch.no_grad()
def latent2image(model, latents, return_type="np"):
    latents = 1 / 0.18215 * latents.detach()
    image = model.decode(latents)["sample"]
    if return_type == "np":
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image


def image_grid(imgs, rows, cols, spacing=20):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new(
        "RGBA",
        size=(cols * w + (cols - 1) * spacing, rows * h + (rows - 1) * spacing),
        color=(255, 255, 255, 0),
    )
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i // rows * (w + spacing), i % rows * (h + spacing)))

    return grid

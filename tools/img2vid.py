import os
import imageio
import numpy as np
import torch
import torchvision
from torchvision.io import read_image
from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # t: n_frames; b: n_batches, if only 1 video then 1
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # outputs is a list of arrays each of shape (H, W, C)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


image_dir = "../metrics/videos/src/add/peter_guitar"
output_dir = "../videos"

if __name__ == "__main__":
    images = [img for img in sorted(os.listdir(image_dir)) if img.endswith(".png")]
    video = torch.stack(
        [read_image(os.path.join(image_dir, img)) for img in images], dim=0
    )  # (F, C, H, W)
    video = video / 127.5 - 1.0  # normalize to [-1, 1]
    videos = rearrange(video, "(b f) c h w -> b c f h w", b=1)

    save_videos_grid(
        videos,
        os.path.join(output_dir, f"{os.path.basename(image_dir)}.mp4"),
        rescale=True,
    )

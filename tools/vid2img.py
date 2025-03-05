import os
import argparse
import torch
import numpy as np
import torchvision
from einops import rearrange
from PIL import Image
import decord

decord.bridge.set_bridge("torch")


def read_video(video_path, video_length, width=512, height=512, frame_rate=None):
    vr = decord.VideoReader(video_path, width=width, height=height)
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(0, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)  # Tensor
    video = rearrange(video, "f h w c -> f c h w")  # uint8, [0, 255]
    video = video / 127.5 - 1.0  # normalize to [-1, 1], torch.float32
    return video


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # t: n_frames; b: n_batches, if only 1 video then 1
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1, 1 -> 0, 1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # save frames
    os.makedirs(path, exist_ok=True)
    for i in range(len(outputs)):
        img = Image.fromarray(outputs[i])
        img.save(f"{path}/%04d.png" % i)


if __name__ == "__main__":
    video_length = 8  # the number of frames
    width, height = 512, 512
    # frame_rate = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="videos/panda_surf.mp4")
    parser.add_argument("--output_dir", type=str, default="videos")
    args = parser.parse_args()

    video = read_video(
        video_path=args.video_path,
        video_length=video_length,
        width=width,
        height=height,
        # frame_rate=frame_rate,
    )
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, args.output_dir, rescale=True)

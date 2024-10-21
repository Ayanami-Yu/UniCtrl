import argparse
import os

import torch
from imageio import mimsave
from pytorch_lightning import seed_everything

from ctrl_video.ctrl_video_zero_pipeline import CtrlVideoZeroPipeline


def dummy(images, **kwargs):
    return images, False


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/video_zero/samples/")
parser.add_argument("--gpu", type=int, default=0)

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
args = parser.parse_args()

src_start, src_inc, src_n = (0.1, 0.1, 31) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.1, 0.1, 41) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "a woman is walking in the rain",
        "a woman is walking in the rain and carrying a red handbag",
    ]
    if not args.prompt
    else args.prompt
)

# set device
torch.cuda.set_device(args.gpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# set seed
seed = 0
seed_everything(seed)

# set output path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

# initialize model
# NOTE setting torch_dtype=torch.float16 in from_pretrained will cause error in unet
model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlVideoZeroPipeline.from_pretrained(model_path, safety_checker=dummy).to(
    device
)

# generate the synthesized videos
src_weights = [round(src_start + src_inc * i, 2) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 2) for i in range(int(tgt_n))]

# TODO provide pre-generated latents
for w_src in src_weights:
    for w_tgt in tgt_weights:
        images = model(
            prompts,
            guidance_scale=7.5,
            use_plain_cfg=False,
            w_src=w_src,
            w_tgt=w_tgt,
            guidance_type="static",
            w_tgt_ctrl_type="static",
            t_ctrl_start=None,
        ).images
        video = [(img * 255).astype("uint8") for img in images]

        # no need to makedirs when no result has been generated
        os.makedirs(out_dir, exist_ok=True)
        mimsave(os.path.join(out_dir, f"{w_src}_{w_tgt}.mp4"), video, fps=4)
        print("Syntheiszed video is saved in", out_dir)

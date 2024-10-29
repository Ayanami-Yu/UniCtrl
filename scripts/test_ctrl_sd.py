import argparse
import os

import torch
from pytorch_lightning import seed_everything
from torchvision import transforms
from torchvision.utils import save_image
from math import cos, sin

from ctrl_image.ctrl_sd_pipeline import CtrlSDPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/sd/samples/")

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
parser.add_argument("--w_src_ctrl_type", type=str, default="static")
parser.add_argument("--w_tgt_ctrl_type", type=str, default="static")

# w_src = scale * cos(theta), w_tgt = scale * sin(theta)
# used to fix the scale applied to the aggregated noise
# theta shoule be in range(0, pi / 2 = 1.5707963267948966)
parser.add_argument("--scale", type=float, default=None)
parser.add_argument("--theta_params", nargs="+", type=float, default=None)
parser.add_argument("--ctrl_mode", type=str, default="add")
parser.add_argument("--removal_version", type=int, default=2)
args = parser.parse_args()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set ctrl params
src_start, src_inc, src_n = (1.3, 0.1, 2) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 31) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "Catwoman holding a sniper rifle",
        "Catwoman holding a sniper rifle and wearing a hat",
    ]
    if not args.prompt
    else args.prompt
)

# set seed
seed = 0
seed_everything(seed)

# set output path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

# initialize model
model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlSDPipeline.from_pretrained(model_path).to(device)

# initialize the noisy latents
# NOTE torch.Generator will produce different results if called for multiple
# times even when the seed is fixed, so the intial noisy latents have to be
# fixed and pre-generated
start_code = torch.randn([1, 4, 64, 64], device=device)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# generate the synthesized images
transform = transforms.ToTensor()

if args.scale is not None:
    x_start, x_inc, x_n = args.theta_params

    for i in range(int(x_n)):
        w_src = round(args.scale * cos(x_start + x_inc * i), 4)
        w_tgt = round(args.scale * sin(x_start + x_inc * i), 4)

        image = model(
            prompts,
            guidance_scale=7.5,
            latents=start_code,
            w_src=w_src,
            w_tgt=w_tgt,
            guidance_type="static",
            w_tgt_ctrl_type="static",
            t_ctrl_start=None,
            ctrl_mode=args.ctrl_mode,
            removal_version=args.removal_version,
        ).images
        image = [transform(img) for img in image]

        # no need to makedirs when no result has been generated
        os.makedirs(out_dir, exist_ok=True)
        save_image(
            image,
            os.path.join(out_dir, f"{w_src:.4f}_{w_tgt:.4f}.png"),
        )
        print("Synthesized images are saved in", out_dir)

else:
    src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
    tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

    for w_src in src_weights:
        for w_tgt in tgt_weights:
            image = model(
                prompts,
                guidance_scale=7.5,
                latents=start_code,
                w_src=w_src,
                w_tgt=w_tgt,
                w_src_ctrl_type=args.w_src_ctrl_type,
                w_tgt_ctrl_type=args.w_tgt_ctrl_type,
                guidance_type="static",
                t_ctrl_start=None,
                ctrl_mode=args.ctrl_mode,
                removal_version=args.removal_version,
            ).images
            image = [transform(img) for img in image]

            # no need to makedirs when no result has been generated
            os.makedirs(out_dir, exist_ok=True)
            save_image(
                image,
                os.path.join(out_dir, f"{w_src}_{w_tgt}.png"),
            )
            print("Synthesized images are saved in", out_dir)

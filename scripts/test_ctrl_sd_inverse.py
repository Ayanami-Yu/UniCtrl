import argparse
import os

import torch
from pytorch_lightning import seed_everything
from torchvision import transforms
from torchvision.utils import save_image

from ctrl_image.ctrl_sd_pipeline import CtrlSDPipeline
from ctrl_utils.inversion import NullInversion

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/sd/inversion/")

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
parser.add_argument("--w_src_ctrl_type", type=str, default="static")
parser.add_argument("--w_tgt_ctrl_type", type=str, default="static")

parser.add_argument("--ctrl_mode", type=str, default="add")
parser.add_argument("--removal_version", type=int, default=2)
args = parser.parse_args()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set ctrl params
src_start, src_inc, src_n = (1.0, 0.1, 1) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.0, 0.5, 3) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "A cat wearing sunglasses and working as a lifeguard at a pool",
        "A cat wearing sunglasses and a hat, working as a lifeguard at a pool",
    ]
    if not args.prompt
    else args.prompt
)

args.prompt

# set seed
seed = 42
seed_everything(seed)

# set output path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

# initialize models
model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlSDPipeline.from_pretrained(model_path).to(device)
null_inversion = NullInversion(
    model, device=device, num_ddim_steps=50, guidance_scale=7.5
)

# generate the synthesized images
transform = transforms.ToTensor()

src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

# perform null-text inversion
src_img_path = "metrics/images/tgt/sd/add/cat_sunglasses_1.0_2.0.png"
(img_gt, img_enc), x_t, uncond_embeddings = null_inversion.invert(
    src_img_path, prompts[0], offsets=(0, 0, 0, 0), verbose=True
)

for w_src in src_weights:
    for w_tgt in tgt_weights:
        image = model(
            prompts,
            guidance_scale=7.5,
            latents=img_enc,   # TODO
            negative_prompt_embeds=uncond_embeddings,
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

        # document the configs
        if not os.path.isfile(f"{out_dir}/configs.txt"):
            with open(os.path.join(out_dir, "configs.txt"), "w") as f:
                f.write(f"seed: {seed}\n")
                f.write(f"prompts: {args.prompt}\n")
                f.write(f"ctrl_mode: {args.ctrl_mode}\n")
                f.write(f"removal_version: {args.removal_version}\n")
                f.write(f"w_tgt_ctrl_type: {args.w_tgt_ctrl_type}\n")
                f.write(f"src_weights: {src_weights}\n")
                f.write(f"tgt_weights: {tgt_weights}\n")
        save_image(
            image,
            os.path.join(out_dir, f"{w_src}_{w_tgt}.png"),
        )
        print("Synthesized images are saved in", out_dir)

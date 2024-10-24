import argparse
import os

import torch
from diffusers.utils import export_to_gif
from pytorch_lightning import seed_everything

from ctrl_3d.ctrl_shap_e_pipeline import CtrlShapEPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/shap_e/samples/")

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
args = parser.parse_args()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set ctrl params
src_start, src_inc, src_n = (0.9, 0.1, 2) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 16) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "a chair",
        "a chair next to a desk",
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

# load pipeline
pipe = CtrlShapEPipeline.from_pretrained(
    "openai/shap-e", torch_dtype=torch.float16, variant="fp16"
)
pipe = pipe.to(device)

# generate synthesized 3d assets
src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

for w_src in src_weights:
    for w_tgt in tgt_weights:
        images = pipe(
            prompts,
            guidance_scale=15.0,
            num_inference_steps=64,
            frame_size=256,
            use_plain_cfg=False,
            guidance_type="static",
            w_src=w_src,
            w_tgt=w_tgt,
            w_src_ctrl_type="static",
            w_tgt_ctrl_type="static",
            t_ctrl_start=None,
        ).images

        # no need to makedirs when no result has been generated
        os.makedirs(out_dir, exist_ok=True)
        export_to_gif(images[0], os.path.join(out_dir, f"{w_src}_{w_tgt}.gif"))
        print("Synthesized result is saved in", out_dir)

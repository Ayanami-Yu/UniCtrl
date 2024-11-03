import argparse
import os

import torch
from diffusers import DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from pytorch_lightning import seed_everything

from ctrl_video.ctrl_animatediff_pipeline import CtrlAnimateDiffPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="../exp/animatediff/samples/")
parser.add_argument("--gpu", type=int, default=0)

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
parser.add_argument("--w_src_ctrl_type", type=str, default="static")
parser.add_argument("--w_tgt_ctrl_type", type=str, default="static")

parser.add_argument("--ctrl_mode", type=str, default="add")
parser.add_argument("--removal_version", type=int, default=2)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_frames", type=int, default=8)
parser.add_argument("--save_as_images", default=False, action="store_true")
args = parser.parse_args()

# set device
torch.cuda.set_device(args.gpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

src_start, src_inc, src_n = (0.9, 0.1, 2) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 16) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "Pikachu taking a selfie",
        "Pikachu taking a selfie, with a red scarf around its neck",
    ]
    if not args.prompt
    else args.prompt
)

# NOTE len(negative_prompts) should match len(prompts),
# with the former used for the uncond embeddings in CFG
negative_prompts = [
    "bad quality, worse quality",
    "bad quality, worse quality",
]

# set seed
seed_everything(args.seed)

# set output path
# NOTE makedirs right after sample count, otherwise output folders might conflict
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

# load the motion adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
)

# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = CtrlAnimateDiffPipeline.from_pretrained(
    model_id, motion_adapter=adapter, torch_dtype=torch.float16
)

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# initialize the noisy latents
# (batch_size, num_channel, num_frames, height, width)
# NOTE AnimateDiff uses float16 for prompt_embeds
start_code = torch.randn([1, 4, 16, 64, 64], device=device, dtype=torch.float16)

# generate the synthesized videos
src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

for w_src in src_weights:
    for w_tgt in tgt_weights:
        output = pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            # num_frames=16,
            num_frames=args.num_frames,
            guidance_scale=7.5,
            num_inference_steps=25,
            # generator=torch.Generator("cpu").manual_seed(seed),
            latents=start_code,
            use_plain_cfg=False,
            w_src=w_src,
            w_tgt=w_tgt,
            guidance_type="static",
            w_src_ctrl_type=args.w_src_ctrl_type,
            w_tgt_ctrl_type=args.w_tgt_ctrl_type,
            t_ctrl_start=None,
            ctrl_mode=args.ctrl_mode,
            removal_version=args.removal_version,
        )
        frames = output.frames[0]

        # document the configs
        if not os.path.isfile(os.path.join(out_dir, "configs.txt")):
            with open(os.path.join(out_dir, "configs.txt"), "w") as f:
                f.write(f"seed: {args.seed}\n")
                f.write(f"prompts: {args.prompt}\n")
                f.write(f"ctrl_mode: {args.ctrl_mode}\n")
                f.write(f"removal_version: {args.removal_version}\n")
                f.write(f"w_tgt_ctrl_type: {args.w_tgt_ctrl_type}\n")
                f.write(f"src_weights: {src_weights}\n")
                f.write(f"tgt_weights: {tgt_weights}\n")
        if not args.save_as_images:
            export_to_gif(frames, os.path.join(out_dir, f"{w_src}_{w_tgt}.gif"))
        else:
            save_path = os.path.join(out_dir, f"{w_src}_{w_tgt}")
            os.makedirs(save_path, exist_ok=True)
            for i in range(len(frames)):
                frames[i].save(os.path.join(save_path, f"{i}.png"))

        print("Synthesized video is saved in", out_dir)

import argparse
import os

import torch
from diffusers import DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from pytorch_lightning import seed_everything

from ctrl_video.ctrl_animatediff_pipeline import CtrlAnimateDiffPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/animatediff/samples/")
parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--gpu", type=str, default="7")

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
args = parser.parse_args()

# set visible GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# set device
torch.cuda.set_device(args.gpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

src_start, src_inc, src_n = (0.9, 0.1, 2) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 16) if not args.tgt_params else args.tgt_params
prompts = (
    [
        "a silver wolf is running",
        "a silver wolf is running after a golden eagle",
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
seed = 0
seed_everything(seed)

# set output path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

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

# inference the synthesized video
src_weights = [round(src_start + src_inc * i, 2) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 2) for i in range(int(tgt_n))]

for w_src in src_weights:
    for w_tgt in tgt_weights:
        output = pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(seed),
            use_plain_cfg=False,
            w_src=w_src,
            w_tgt=w_tgt,
            guidance_type="static",
            w_tgt_ctrl_type="static",
            t_ctrl_start=None,
        )
        frames = output.frames[0]

        # no need to makedirs when no result has been generated
        os.makedirs(out_dir, exist_ok=True)
        export_to_gif(frames, os.path.join(out_dir, f"{w_src}_{w_tgt}.gif"))
        print("Syntheiszed video is saved in", out_dir)

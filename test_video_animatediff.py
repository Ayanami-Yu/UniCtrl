import os
import torch
import argparse

from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from pytorch_lightning import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", nargs="+", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./exp/video/animatediff/")
parser.add_argument("--gpu", type=int, default=0)

# weight_start, weight_inc, weight_n
parser.add_argument("--src_params", nargs="+", type=float, default=None)
parser.add_argument("--tgt_params", nargs="+", type=float, default=None)
args = parser.parse_args()

src_start, src_inc, src_n = (0.1, 0.1, 31) if not args.src_params else args.src_params
tgt_start, tgt_inc, tgt_n = (0.1, 0.1, 41) if not args.tgt_params else args.tgt_params
prompts = (
    [
        (
            "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
            "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
            "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
            "golden hour, coastal landscape, seaside scenery"
        )
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

# load the motion adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
)

# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(
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
            negative_prompt="bad quality, worse quality",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        frames = output.frames[0]
        export_to_gif(frames, "animation.gif")

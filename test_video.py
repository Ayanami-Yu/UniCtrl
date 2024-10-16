import os
import torch

from imageio import mimsave
from torchvision.utils import save_image
from torchvision import transforms
from pytorch_lightning import seed_everything
from diffusers.utils import export_to_video
from ctrl_video.video_pipeline import VideoPipeline


def dummy(images, **kwargs):
    return images, False


src_start, src_inc, src_n = 0.9, 0.1, 1
tgt_start, tgt_inc, tgt_n = 0.1, 0.1, 1
prompts = ["a cat is running on a road", "a cat with a pair of wings is running on a road"]  # TODO error when two prompts

# set device
torch.cuda.set_device(2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# set seed
seed = 0
seed_everything(seed)
# generator = torch.Generator(device=device)


# set output path
out_dir = "./exp/video/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

# initialize model
# NOTE setting torch_dtype=torch.float16 in from_pretrained will cause error in unet
model_path = "/mnt/hdd1/hongyu/models/stable-diffusion-2-1-base"
model = VideoPipeline.from_pretrained(model_path, safety_checker=dummy).to(device)

# initialize the noise map
# start_code = torch.randn([1, 4, 64, 64], device=device)
# start_code = start_code.expand(len(prompts), -1, -1, -1)

# inference the synthesized video
src_weights = [round(src_start + src_inc * i, 2) for i in range(src_n)]
tgt_weights = [round(tgt_start + tgt_inc * i, 2) for i in range(tgt_n)]

for w_src in src_weights:
    for w_tgt in tgt_weights:
        images = model(
            prompts,
            # latents=start_code,
            # generator=generator,
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

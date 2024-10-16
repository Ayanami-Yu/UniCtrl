import os
import torch
from torchvision.utils import save_image
from pytorch_lightning import seed_everything


from custom_pipeline import CustomPipeline
from attention_utils import register_attention_editor_diffusers, AttentionBase

src_start, src_inc, src_n = 0.9, 0.1, 1
tgt_start, tgt_inc, tgt_n = 0.5, 0.1, 11
prompts = [
    "an astronaut riding a horse",
    "an astronaut riding a horse and holding a Gatling gun",
]

# set seed
seed = 0
seed_everything(seed)

# set device
torch.cuda.set_device(1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# set output path
out_dir = "./exp/add/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

# initialize model
model_path = "/mnt/hdd1/hongyu/models/stable-diffusion-2-1-base"
model = CustomPipeline.from_pretrained(model_path).to(device)

# initialize the noise map
start_code = torch.randn([1, 4, 64, 64], device=device)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# use original attention
editor = AttentionBase()
register_attention_editor_diffusers(model, editor)

# inference the synthesized image
src_weights = [round(src_start + src_inc * i, 2) for i in range(src_n)]
tgt_weights = [round(tgt_start + tgt_inc * i, 2) for i in range(tgt_n)]

for w_src in src_weights:
    for w_tgt in tgt_weights:
        image = model(
            prompts,
            latents=start_code,
            guidance_scale=7.5,
            w_src=w_src,
            w_tgt=w_tgt,
            guidance_type="static",
            t_ctrl_start=630,
        )

        # save the synthesized image
        save_image(
            image,
            os.path.join(out_dir, f"{w_src}_{w_tgt}.png"),
        )
        print("Syntheiszed images are saved in", out_dir)

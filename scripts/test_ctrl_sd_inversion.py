import os
import torch

from pytorch_lightning import seed_everything
from PIL import Image
from ctrl_image.ctrl_sd_direct_inversion_pipeline import CtrlSDDirectInversionPipeline
from ctrl_utils.image_utils import image_grid


image_path = "metrics/images/src/default/add/dog_soccer_1.0_0.0.png"
out_dir = "exp/sd/real/dog_soccer"

mode = "add"
src_start, src_inc, src_n = (1.0, 0.1, 1)
tgt_start, tgt_inc, tgt_n = (0.3, 0.2, 12)

prompts = [
    "a cute pomeranian dog is playing",
    "a cute pomeranian dog is playing with a soccer ball",
]

torch.cuda.set_device(2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 42
seed_everything(seed)

os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")

model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlSDDirectInversionPipeline.from_pretrained(model_path).to(device)

src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

image_src = Image.open(image_path)

for scale in range(12, 15, 2):  # TODO check effect of CFG
    scale = scale / 10
    for w_src in src_weights:
        for w_tgt in tgt_weights:
            image_rec, image_edit = model(
                prompts,
                guidance_scale=scale,
                w_src=w_src,
                w_tgt=w_tgt,
                w_src_ctrl_type="static",
                w_tgt_ctrl_type="static",
                ctrl_mode=mode,
                image_path=image_path,
                do_direct_inversion=True,
            )
            os.makedirs(out_dir, exist_ok=True)

            image = image_grid((image_src, image_rec, image_edit), rows=1, cols=3)
            # image.save(os.path.join(out_dir, f"{w_src}_{w_tgt}.png"))
            image.save(os.path.join(out_dir, f"{w_src}_{w_tgt}_scale_{scale}.png"))

print("Synthesized images are saved in", out_dir)

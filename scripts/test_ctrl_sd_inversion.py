import torch

from pytorch_lightning import seed_everything
from PIL import Image
from ctrl_image.ctrl_sd_inversion_pipeline import CtrlSDInversionPipeline
from ctrl_utils.image_utils import image_grid


image_path = "metrics/images/src/default/rm/witch_book_1.0_-0.22.png"
name = "witch_book"

mode = "remove"
w_src, w_tgt = 1.0, -0.22
prompts = [
    "a witch reading a large open book, fantasy, anime",
    "a large open book",
]

torch.cuda.set_device(2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 42
seed_everything(seed)

model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlSDInversionPipeline.from_pretrained(model_path).to(device)

image_rec, image_edit = model(
    prompts,
    guidance_scale=7.5,
    w_src=w_src,
    w_tgt=w_tgt,
    w_src_ctrl_type="static",
    w_tgt_ctrl_type="cosine",
    ctrl_mode=mode,
    image_path=image_path,
)
image_src = Image.open(image_path)
image = image_grid((image_src, image_rec, image_edit), rows=1, cols=3)
image.save(f"{name}.png")

import torch
from diffusers.utils import export_to_gif

from ctrl_3d.ctrl_shap_e_pipeline import CtrlShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = CtrlShapEPipeline.from_pretrained(
    "openai/shap-e", torch_dtype=torch.float16, variant="fp16"
)
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["A firecracker", "A birthday cupcake"]

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

export_to_gif(images[0], "firecracker_3d.gif")
export_to_gif(images[1], "cake_3d.gif")

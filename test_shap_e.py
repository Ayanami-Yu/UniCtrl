import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained(
    "openai/shap-e", torch_dtype=torch.float16, variant="fp16"
)
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = [
    "a horse galloping on the street, best quality",
    "a horse galloping on the street with a girl riding on it, best quality",
]

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

export_to_gif(images[0], "src.gif")
export_to_gif(images[1], "tgt.gif")

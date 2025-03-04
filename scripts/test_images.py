import os
import yaml
import torch

from pytorch_lightning import seed_everything
from collections import namedtuple
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ctrl_image.ctrl_sd_pipeline import CtrlSDPipeline
from metrics.clip_utils import DirectionalSimilarity


# set input and output paths
out_dir = "exp/sd/pie"
yaml_path = "pie_prompts_modified.yaml"

# set parameters
seed = 206096096
device_idx = 7

modes = ["style"]  # available modes: add, rm, style
w_tgt_ctrl_type = "cosine"

# prepare for generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device_idx)

seed_everything(seed)
os.makedirs(out_dir, exist_ok=True)

model_path = "stabilityai/stable-diffusion-2-1-base"
model = CtrlSDPipeline.from_pretrained(model_path).to(device)

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

Result = namedtuple("Result", ["image", "w_src", "w_tgt"])
Pair = namedtuple("Pair", ["res_src", "res_tgt", "clip_dir"])

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
dir_similarity = DirectionalSimilarity(
    tokenizer, text_encoder, image_processor, image_encoder, device
)

for mode in modes:
    if mode == "add" or mode == "style":
        ctrl_mode = "add"
    else:
        ctrl_mode = "remove"

    for case in data[mode].values():
        src_start, src_inc, src_n = (1.0, 0.1, 1)
        if mode == "rm":
            tgt_start, tgt_inc, tgt_n = (
                (-1.0, 0.05, 44) if w_tgt_ctrl_type == "static" else (-1.0, 0.1, 36)
            )
        elif mode == "add" and w_tgt_ctrl_type == "static":
            tgt_start, tgt_inc, tgt_n = (0.0, 0.05, 44)
        else:
            tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 33)

        src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
        tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

        prompts = [
            case["src_prompt"],
            case["tgt_prompt"]["change"] if mode == 'rm' else case['tgt_prompt'],
        ]
        cur_dir = os.path.join(
            out_dir,
            f"{mode}_{w_tgt_ctrl_type}_seed_{seed}",
            prompts[0] + " | " + prompts[1],
        )
        os.makedirs(cur_dir, exist_ok=True)

        start_code = torch.randn([1, 4, 64, 64], device=device)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        for w_src in src_weights:
            results = []
            pairs = []
            for w_tgt in tgt_weights:
                image = model(
                    prompts,
                    guidance_scale=7.5,
                    latents=start_code,
                    w_src=w_src,
                    w_tgt=w_tgt,
                    w_src_ctrl_type="static",
                    w_tgt_ctrl_type=w_tgt_ctrl_type,
                    ctrl_mode=ctrl_mode,
                ).images[0]

                # document the configs
                configs = {
                    "seed": seed,
                    "prompts": prompts,
                    "ctrl_mode": mode,
                    "w_tgt_ctrl_type": w_tgt_ctrl_type,
                    "src_weights": src_weights,
                    "tgt_weights": tgt_weights,
                }
                yaml_path = os.path.join(cur_dir, "configs.yaml")
                if not os.path.isfile(yaml_path):
                    with open(yaml_path, "w") as f:
                        yaml.dump(configs, f, default_flow_style=False)

                results.append(Result(image, w_src, w_tgt))

            results.sort(key=lambda x: x.w_tgt)
            for i in range(len(results)):
                if (ctrl_mode == "add" and results[i].w_tgt > 1.5) or (
                    ctrl_mode == "remove" and results[i].w_tgt > 1.0
                ):
                    break

                for j in range(i + 1, len(results)):
                    clip_dir = dir_similarity(
                        results[i].image,
                        results[j].image,
                        prompts[0],
                        (
                            prompts[1]
                            if ctrl_mode == "add"
                            else case["tgt_prompt"]["default"]
                        ),
                    )
                    pairs.append(
                        Pair(results[i], results[j], float(clip_dir.detach().cpu()))
                    )

            pairs.sort(key=lambda x: x.clip_dir, reverse=True)
            for i in range(6):
                src, tgt = pairs[i].res_src, pairs[i].res_tgt
                src.image.save(
                    os.path.join(cur_dir, f"{i}_src_{src.w_src}_{src.w_tgt}.png")
                )
                tgt.image.save(
                    os.path.join(cur_dir, f"{i}_tgt_{tgt.w_src}_{tgt.w_tgt}.png")
                )

        print("Synthesized images are saved in", cur_dir)

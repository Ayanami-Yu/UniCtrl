import os
import yaml
import torch

from pytorch_lightning import seed_everything
from diffusers import DDIMScheduler, MotionAdapter
from collections import namedtuple
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from ctrl_video.ctrl_animatediff_pipeline import CtrlAnimateDiffPipeline
from metrics.clip_utils import DirectionalSimilarity


# set input and output paths
out_dir = "exp/animatediff/pie"
yaml_path = "docs/prompts_video_v1.yaml"

# set parameters
seed = 1131219402
# device_idx = 4

modes = ["rm"]  # available modes: add, rm, style
w_tgt_ctrl_type = "static"

# prepare for generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device_idx)

os.makedirs(out_dir, exist_ok=True)

model_path = "stabilityai/stable-diffusion-2-1-base"

num_frames = 8
negative_prompts = [
    "bad quality, worse quality",
    "bad quality, worse quality",
]

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

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

Result = namedtuple("Result", ["video", "w_src", "w_tgt"])
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
                (-1.0, 0.1, 33) if w_tgt_ctrl_type == "static" else (-1.0, 0.1, 36)
            )
        elif w_tgt_ctrl_type == "static":
            tgt_start, tgt_inc, tgt_n = (0.0, 0.1, 33)
        else:
            tgt_start, tgt_inc, tgt_n = (0.0, 0.15, 40)  # for cosine

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

        seed_everything(seed)
        start_code = torch.randn(
            [1, 4, num_frames, 64, 64], device=device, dtype=torch.float16
        )

        for w_src in src_weights:
            results = []
            pairs = []
            for w_tgt in tgt_weights:
                output = pipe(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    num_frames=num_frames,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    latents=start_code,
                    w_src=w_src,
                    w_tgt=w_tgt,
                    w_tgt_ctrl_type=w_tgt_ctrl_type,
                    ctrl_mode=ctrl_mode,
                )
                frames = output.frames[0]

                # document the configs
                tgt_prompt_default = case['tgt_prompt']['default'] if mode == 'rm' else case['tgt_prompt']
                configs = {
                    "seed": seed,
                    "prompts": prompts,
                    "tgt_prompt_default": tgt_prompt_default,
                    "ctrl_mode": mode,
                    "w_tgt_ctrl_type": w_tgt_ctrl_type,
                    "src_weights": src_weights,
                    "tgt_weights": tgt_weights,
                }
                yaml_path = os.path.join(cur_dir, "configs.yaml")
                if not os.path.isfile(yaml_path):
                    with open(yaml_path, "w") as f:
                        yaml.dump(configs, f, default_flow_style=False)

                results.append(Result(frames, w_src, w_tgt))

            results.sort(key=lambda x: x.w_tgt)
            for i in range(len(results)):
                if (ctrl_mode == "add" and results[i].w_tgt > 1.5) or (
                    ctrl_mode == "remove" and results[i].w_tgt > 1.0
                ):
                    break

                # to save computation we only calculate CLIP directional similarity
                # for the first frames to select the best
                for j in range(i + 1, len(results)):
                    clip_dir = dir_similarity(
                        results[i].video[0],
                        results[j].video[0],
                        prompts[0],
                        tgt_prompt_default,
                    )
                    pairs.append(
                        Pair(results[i], results[j], float(clip_dir.detach().cpu()))
                    )

            pairs.sort(key=lambda x: x.clip_dir, reverse=True)
            for i in range(6):
                src, tgt = pairs[i].res_src, pairs[i].res_tgt
                save_dir_src = os.path.join(cur_dir, f"{i}_src_{src.w_src}_{src.w_tgt}")
                os.makedirs(save_dir_src, exist_ok=True)
                for j in range(len(src.video)):
                    src.video[j].save(f'{save_dir_src}/%05d.png' % j)
                save_dir_tgt = os.path.join(cur_dir, f"{i}_tgt_{tgt.w_src}_{tgt.w_tgt}")
                os.makedirs(save_dir_tgt, exist_ok=True)
                for j in range(len(tgt.video)):
                    tgt.video[j].save(f'{save_dir_tgt}/%05d.png' % j)

        print("Synthesized images are saved in", cur_dir)

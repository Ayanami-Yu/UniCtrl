import os
from typing import Optional

import imageio
import kiui
import numpy as np
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import tyro
from kiui.cam import orbit_camera
from kiui.op import recenter
from pytorch_lightning import seed_everything
from safetensors.torch import load_file

from ctrl_3d.args_lgm import AllConfigs
from ctrl_3d.ctrl_mvdream_pipeline import CtrlMVDreamPipeline
from ctrl_3d.LGM.core.models import LGM

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith("safetensors"):
        ckpt = load_file(opt.resume, device="cpu")
    else:
        ckpt = torch.load(opt.resume, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint from {opt.resume}")
else:
    print(f"[WARN] model randomly initialized, are you sure?")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load dreams
pipe = CtrlMVDreamPipeline.from_pretrained(
    "ashawkey/mvdream-sd2.1-diffusers",  # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()


# process function
def process(
    prompt,
    prompt_neg="",
    input_elevation=0,
    input_num_steps=30,
    out_dir="exp/lgm/",
    latents=None,
    use_plain_cfg=False,
    guidance_type: str = "static",
    w_src=1.0,
    w_tgt=1.0,
    w_src_ctrl_type: str = "static",
    w_tgt_ctrl_type: str = "static",
    t_ctrl_start: Optional[int] = None,
    ctrl_mode: str = "add",
    removal_version: int = 1,
    save_as_images: bool = False,
    save_images_interval: int = 1,
):
    # text-conditioned
    mv_image_uint8 = pipe(
        prompt,
        negative_prompt=prompt_neg,
        num_inference_steps=input_num_steps,
        guidance_scale=7.5,
        elevation=input_elevation,
        latents=latents,
        use_plain_cfg=use_plain_cfg,
        guidance_type=guidance_type,
        w_src=w_src,
        w_tgt=w_tgt,
        w_src_ctrl_type=w_src_ctrl_type,
        w_tgt_ctrl_type=w_tgt_ctrl_type,
        t_ctrl_start=t_ctrl_start,
        ctrl_mode=ctrl_mode,
        removal_version=removal_version,
    )
    mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)

    # bg removal
    mv_image = []
    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover)  # [H, W, 4]
        # to white bg
        image = image.astype(np.float32) / 255
        image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)

    mv_image_grid = np.concatenate(
        [
            np.concatenate([mv_image[1], mv_image[2]], axis=1),
            np.concatenate([mv_image[3], mv_image[0]], axis=1),
        ],
        axis=0,
    )

    # generate gaussians
    input_image = np.stack(
        [mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0
    )  # [4, 256, 256, 3], float32
    input_image = (
        torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device)
    )  # [4, 3, 256, 256]
    input_image = F.interpolate(
        input_image,
        size=(opt.input_size, opt.input_size),
        mode="bilinear",
        align_corners=False,
    )
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # for debug
    # from torchvision.utils import save_image
    # save_image(input_image, "sample.jpg")

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(
        0
    )  # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)

        # save gaussians
        # model.gs.save_ply(gaussians, os.path.join(out_dir, f"{w_src}_{w_tgt}.ply"))

        # render 360 video
        images = []
        elevation = 0
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):

                cam_poses = (
                    torch.from_numpy(
                        orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(
                    gaussians,
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                    scale_modifier=scale,
                )["image"]
                images.append(
                    (
                        image.squeeze(1)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .float()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):

                cam_poses = (
                    torch.from_numpy(
                        orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

                image = model.gs.render(
                    gaussians,
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                    scale_modifier=1,
                )["image"]
                images.append(
                    (
                        image.squeeze(1)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .float()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )

        if save_as_images:
            cur_dir = os.path.join(out_dir, f"{w_src}_{w_tgt}")
            os.makedirs(cur_dir, exist_ok=True)

            images = images[0::save_images_interval]
            img_paths = [os.path.join(cur_dir, f"{i}.png") for i in range(len(images))]
            for img, path in zip(images, img_paths):
                imageio.imwrite(path, np.squeeze(img, axis=0), format="png")
        else:
            output_video_path = os.path.join(out_dir, f"{w_src}_{w_tgt}.mp4")
            images = np.concatenate(images, axis=0)
            imageio.mimwrite(output_video_path, images, fps=30)

        print("Synthesized result is saved in", out_dir)

    return mv_image_grid


# set seed
seed = 0
seed_everything(seed)

# whether to save as images or a video
# NOTE LGM will generate 180 images for a video by default
save_as_images = True
save_images_interval = 22

# set output path
out_dir = opt.workspace
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

# initialize the noisy latents
start_code = torch.randn([4, 4, 32, 32], device=device, dtype=torch.float16)

negative_prompts = [
    "",
    "",
]

src_start, src_inc, src_n = opt.src_params
tgt_start, tgt_inc, tgt_n = opt.tgt_params

src_weights = [round(src_start + src_inc * i, 4) for i in range(int(src_n))]
tgt_weights = [round(tgt_start + tgt_inc * i, 4) for i in range(int(tgt_n))]

# document the configs
with open(os.path.join(out_dir, "configs.txt"), "w") as f:
    f.write(f"seed: {seed}\n")
    f.write(f"prompts: {opt.prompts}\n")
    f.write(f"ctrl_mode: {opt.ctrl_mode}\n")
    f.write(f"removal_version: {opt.removal_version}\n")
    f.write(f"w_tgt_ctrl_type: {opt.w_tgt_ctrl_type}\n")
    f.write(f"src_weights: {src_weights}\n")
    f.write(f"tgt_weights: {tgt_weights}\n")

for w_src in src_weights:
    for w_tgt in tgt_weights:
        process(
            prompt=opt.prompts,
            prompt_neg=negative_prompts,
            out_dir=out_dir,
            latents=start_code,
            use_plain_cfg=False,
            guidance_type="static",
            w_src=w_src,
            w_tgt=w_tgt,
            w_src_ctrl_type=opt.w_src_ctrl_type,
            w_tgt_ctrl_type=opt.w_tgt_ctrl_type,
            t_ctrl_start=None,
            ctrl_mode=opt.ctrl_mode,
            removal_version=opt.removal_version,
            save_as_images=save_as_images,
            save_images_interval=save_images_interval,
        )

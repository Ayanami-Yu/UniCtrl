#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
from random import randint

import imageio
import torch
from ctrl_3d.LucidDreamer.arguments import (
    GuidanceParams,
)
from ctrl_3d.LucidDreamer.gaussian_renderer import network_gui, render
from ctrl_3d.LucidDreamer.scene import GaussianModel, Scene
from ctrl_3d.LucidDreamer.train import (
    prepare_output_and_logger,
    training_report,
    video_inference,
)
from ctrl_3d.LucidDreamer.utils.loss_utils import tv_loss
from tqdm import tqdm
from ctrl_3d.args_lucid_dreamer import CtrlParams
from ctrl_3d.sd_lucid_dreamer import StableDiffusionCtrl


def prepare_embeddings(
    guidance: StableDiffusionCtrl, guidance_opt: GuidanceParams, text_prompt: str
):
    embeddings = {}
    embeddings["text"] = guidance.get_text_embeds(text_prompt)
    embeddings["uncond"] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ["front", "side", "back"]:
        embeddings[d] = guidance.get_text_embeds([f"{text_prompt}, {d} view"])
    embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)
    return embeddings


def guidance_setup(guidance_opt: GuidanceParams, ctrl_params: CtrlParams):
    if guidance_opt.guidance != "SD":
        raise ValueError(f"{guidance_opt.guidance} not supported.")

    guidance = StableDiffusionCtrl(
        guidance_opt.g_device,
        guidance_opt.fp16,
        guidance_opt.vram_O,
        guidance_opt.t_range,
        guidance_opt.max_t_range,
        num_train_timesteps=guidance_opt.num_train_timesteps,
        ddim_inv=guidance_opt.ddim_inv,
        textual_inversion_path=guidance_opt.textual_inversion_path,
        LoRA_path=guidance_opt.LoRA_path,
        guidance_opt=guidance_opt,
    )
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False

    embeddings = [
        prepare_embeddings(guidance, guidance_opt, ctrl_params.src_prompt)
    ] + [prepare_embeddings(guidance, guidance_opt, ctrl_params.tgt_prompt)]
    return guidance, embeddings


def training(
    dataset,
    opt,
    pipe,
    gcams,
    guidance_opt,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    save_video,
    ctrl_params: CtrlParams,
):
    assert not guidance_opt.perpneg, "Prompt ctrl with Perp-Neg not implemented"

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, use_tensorboard=False)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    save_folder = os.path.join(dataset._model_path, "train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("train_process is in :", save_folder)

    # set up pretrained diffusion models and text_embedings
    guidance, embeddings = guidance_setup(guidance_opt, ctrl_params)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path, "process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)
        process_view_points = scene.getCircleVideoCameras(
            batch_size=opt.pro_frames_num, render45=opt.pro_render_45
        ).copy()
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []

    # controlnet
    use_control_net = False

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)

        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Progressively relaxing view range
        if not opt.use_progressive:
            if (
                iteration >= opt.progressive_view_iter
                and iteration % opt.scale_up_cameras_iter == 0
            ):
                scene.pose_args.fovy_range[0] = max(
                    scene.pose_args.max_fovy_range[0],
                    scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0],
                )
                scene.pose_args.fovy_range[1] = min(
                    scene.pose_args.max_fovy_range[1],
                    scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1],
                )

                scene.pose_args.radius_range[1] = max(
                    scene.pose_args.max_radius_range[1],
                    scene.pose_args.radius_range[1] * opt.scale_up_factor,
                )
                scene.pose_args.radius_range[0] = max(
                    scene.pose_args.max_radius_range[0],
                    scene.pose_args.radius_range[0] * opt.scale_up_factor,
                )

                scene.pose_args.theta_range[1] = min(
                    scene.pose_args.max_theta_range[1],
                    scene.pose_args.theta_range[1] * opt.phi_scale_up_factor,
                )
                scene.pose_args.theta_range[0] = max(
                    scene.pose_args.max_theta_range[0],
                    scene.pose_args.theta_range[0] * 1 / opt.phi_scale_up_factor,
                )

                scene.pose_args.phi_range[0] = max(
                    scene.pose_args.max_phi_range[0],
                    scene.pose_args.phi_range[0] * opt.phi_scale_up_factor,
                )
                scene.pose_args.phi_range[1] = min(
                    scene.pose_args.max_phi_range[1],
                    scene.pose_args.phi_range[1] * opt.phi_scale_up_factor,
                )

                print("scale up theta_range to:", scene.pose_args.theta_range)
                print("scale up radius_range to:", scene.pose_args.radius_range)
                print("scale up phi_range to:", scene.pose_args.phi_range)
                print("scale up fovy_range to:", scene.pose_args.fovy_range)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getRandTrainCameras().copy()

        # Index 0 corresponds to src and 1 to tgt
        # NOTE only used in add_noise_with_cfg where prompt ctrl is not involved,
        # so only one tensor is needed
        text_z_inverse = torch.cat(
            [embeddings[0]["uncond"], embeddings[0]["inverse_text"]], dim=0
        )

        # 1) Render 3D assets
        viewpoint_cams = []
        images = []
        depths = []
        alphas = []
        scales = []
        for _ in range(guidance_opt.C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )
            except:
                viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )

            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
                bg_aug_ratio=dataset.bg_aug_ratio,
                shs_aug_ratio=dataset.shs_aug_ratio,
                scale_aug_ratio=dataset.scale_aug_ratio,
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # 2) Adjust text embeddings
        text_z_ = []
        weights_ = []
        for embs in embeddings:
            for i in range(guidance_opt.C_batch_size):
                azimuth = viewpoint_cams[i].delta_azimuth

                # TODO will interpolation between embeds affect prompt ctrl?
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = embs["front"]
                    end_z = embs["side"]
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = embs["side"]
                    end_z = embs["back"]
                text_z = r * start_z + (1 - r) * end_z

                text_z = torch.cat((embs["uncond"], text_z), dim=0)
                text_z_.append(text_z)

        # Loss
        warm_up_rate = 1.0 - min(iteration / opt.warmup_iter, 1.0)
        _aslatent = False
        if iteration < opt.geo_iter or random.random() < opt.as_latent_ratio:
            _aslatent = True
        if iteration > opt.use_control_net_iter and (
            random.random() < guidance_opt.controlnet_ratio
        ):
            use_control_net = True

        loss = guidance.train_step(
            torch.stack(text_z_, dim=1),
            images,
            pred_depth=depths,
            pred_alpha=alphas,
            grad_scale=guidance_opt.lambda_guidance,
            use_control_net=use_control_net,
            save_folder=save_folder,
            iteration=iteration,
            warm_up_rate=warm_up_rate,
            resolution=(gcams.image_h, gcams.image_w),
            guidance_opt=guidance_opt,
            as_latent=_aslatent,
            embedding_inverse=text_z_inverse,
            use_plain_cfg=False,
            guidance_type=ctrl_params.guidance_type,
            w_src=ctrl_params.w_src,
            w_tgt=ctrl_params.w_tgt,
            w_src_ctrl_type=ctrl_params.w_src_ctrl_type,
            w_tgt_ctrl_type=ctrl_params.w_tgt_ctrl_type,
            t_ctrl_start=ctrl_params.t_ctrl_start,
        )
        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales, dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths)
        # loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

        loss = (
            loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale
        )  # opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(
                        viewpoint_cam_p, gaussians, pipe, background, test=True
                    )
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0)
                    img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
                    img_p = (img_p * 255).round().astype("uint8")
                    pro_img_frames.append(img_p)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in testing_iterations:
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0
                ):  # or (dataset._white_background and iteration == opt.densify_from_iter)
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene._model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    if opt.save_process:
        imageio.mimwrite(
            os.path.join(save_folder_proc, "video_rgb.mp4"),
            pro_img_frames,
            fps=30,
            quality=8,
        )

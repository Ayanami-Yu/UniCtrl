import sys
from argparse import ArgumentParser

import torch

from ctrl_3d.args_lucid_dreamer import CtrlParams
from ctrl_3d.LucidDreamer.arguments import (
    GenerateCamParams,
    GuidanceParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from ctrl_3d.LucidDreamer.gaussian_renderer import network_gui
from ctrl_3d.LucidDreamer.utils.general_utils import safe_state
from ctrl_3d.train_lucid_dreamer import training

if __name__ == "__main__":
    import yaml

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--opt", type=str, default=None)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_ratio", type=int, default=5)
    parser.add_argument("--save_ratio", type=int, default=2)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--w_src_cli", type=float, default=None)
    parser.add_argument("--w_tgt_cli", type=float, default=None)
    parser.add_argument("--workspace_cli", type=str, default=None)
    parser.add_argument("--ctrl_mode_cli", type=str, default=None)
    parser.add_argument("--removal_version_cli", type=int, default=None)

    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)
    cp = CtrlParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        mp.load_yaml(opts.get("ModelParams", None))
        op.load_yaml(opts.get("OptimizationParams", None))
        pp.load_yaml(opts.get("PipelineParams", None))
        gcp.load_yaml(opts.get("GenerateCamParams", None))
        gp.load_yaml(opts.get("GuidanceParams", None))
        cp.load_yaml(opts.get("CtrlParams", None))

        mp.opt_path = args.opt
        args.port = opts["port"]
        args.save_video = opts.get("save_video", True)
        args.seed = opts.get("seed", 0)
        args.device = opts.get("device", "cuda")

        # override device
        gp.g_device = args.device
        mp.data_device = args.device
        gcp.device = args.device

    # save iterations
    test_iter = (
        [1]
        + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)]
        + [op.iterations]
    )
    args.test_iterations = test_iter

    save_iter = [
        k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)
    ] + [op.iterations]
    args.save_iterations = save_iter

    # override the values with the params from command line
    if args.w_src_cli and args.w_tgt_cli:
        cp.w_src, cp.w_tgt = args.w_src_cli, args.w_tgt_cli
    if args.workspace_cli is not None:
        mp.workspace = args.workspace_cli
    if args.ctrl_mode_cli is not None:
        cp.ctrl_mode = args.ctrl_mode_cli
    if args.removal_version_cli is not None:
        cp.removal_version = args.removal_version_cli

    print("Test iter:", args.test_iterations)
    print("Save iter:", args.save_iterations)

    print("Optimizing " + mp._model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        mp,
        op,
        pp,
        gcp,
        gp,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.save_video,
        ctrl_params=cp,
    )

    # All done
    print("\nTraining complete.")

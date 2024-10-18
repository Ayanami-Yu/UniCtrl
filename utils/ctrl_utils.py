import torch
import numpy as np


# TODO should it adapt to different tensors of different shapes?
def get_perpendicular_component(x, y):
    """Get the component of x that is perpendicular to y"""
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum()) / (torch.norm(y) ** 2)) * y


def add_aggregator_v1(delta_noise_pred_src, w_src, delta_noise_pred_tgt, w_tgt):
    return w_src * delta_noise_pred_src + w_tgt * get_perpendicular_component(
        delta_noise_pred_tgt, delta_noise_pred_src
    )


def add_aggregator_v2(delta_noise_pred_src, w_src, delta_noise_pred_tgt, w_tgt):
    return w_src * delta_noise_pred_src + w_tgt * (
        get_perpendicular_component(delta_noise_pred_tgt, delta_noise_pred_src)
        - delta_noise_pred_src
    )


def guidance_weight(t, w0, guidance_type: str, t_total=1000, clamp=4):
    if guidance_type == "static":
        w = w0
    elif guidance_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    elif guidance_type == "cosine":
        w = w0 * (np.cos(np.pi * t / t_total) + 1)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w


def ctrl_weight(t, w0, ctrl_type: str, t_total=1000, clamp=None):
    if ctrl_type == "static":
        w = w0
    elif ctrl_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w

import torch


def get_perpendicular_component(x, y, mode: str):
    """Get the component of x that is perpendicular to y"""
    assert x.shape == y.shape
    if mode == "all":
        return x - ((torch.mul(x, y).sum()) / (torch.norm(y) ** 2)) * y
    elif mode == "channel":
        return (
            x
            - (
                (torch.mul(x, y).sum(dim=(-2, -1), keepdim=True))
                / (torch.norm(y, dim=(-2, -1), keepdim=True) ** 2)
            )
            * y
        )
    elif mode == "latent":
        prod_xy = torch.einsum("...ijk, ...ijk -> ...", x, y)
        prod_yy = torch.einsum("...ijk, ...ijk -> ...", y, y)
        return x - (prod_xy / prod_yy).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * y
    else:
        raise ValueError("Unrecognized mode for perpendicular component")


def add_aggregator_v1(
    delta_noise_pred_src,
    w_src,
    delta_noise_pred_tgt,
    w_tgt,
    mode: str = "latent",
):
    return w_src * delta_noise_pred_src + w_tgt * get_perpendicular_component(
        delta_noise_pred_tgt, delta_noise_pred_src, mode=mode
    )


def add_aggregator_v2(
    delta_noise_pred_src,
    w_src,
    delta_noise_pred_tgt,
    w_tgt,
    mode: str = "latent",
):
    return w_src * delta_noise_pred_src + w_tgt * (
        get_perpendicular_component(
            delta_noise_pred_tgt, delta_noise_pred_src, mode=mode
        )
        - delta_noise_pred_src
    )


def remove_aggregator_v1(
    delta_noise_pred_src,
    w_src,
    delta_noise_pred_tgt,
    w_tgt,
    mode: str = "latent",
):
    """
    Params:
        w_tgt:
            The strength to extract the semantics of target from source, should be
            greater than 0. When w_tgt equals 0, the result will be the perpendicular
            component of delta_noise_pred_src.
    """
    return (
        w_src
        * get_perpendicular_component(
            delta_noise_pred_src, delta_noise_pred_tgt, mode=mode
        )
        - w_tgt * delta_noise_pred_tgt
    )


def remove_aggregator_v2(
    delta_noise_pred_src,
    w_src,
    delta_noise_pred_tgt,
    w_tgt,
    mode: str = "latent",
):
    """
    Params:
        w_tgt:
            The strength to extract the semantics of target from source, should be
            greater than -1. When w_tgt equals -1, the result will be delta_noise_pred_src.
    """
    noise_pred_src_perp = get_perpendicular_component(
        delta_noise_pred_src, delta_noise_pred_tgt, mode=mode
    )
    return w_src * noise_pred_src_perp - w_tgt * (
        delta_noise_pred_src - noise_pred_src_perp
    )


def guidance_weight(t, w0, guidance_type: str, t_total=1000, clamp=4):
    if guidance_type == "static":
        w = w0
    elif guidance_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    elif guidance_type == "cosine":
        w = w0 * (torch.cos(torch.pi * t / t_total) + 1)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w


def ctrl_weight(t, w0, ctrl_type: str, t_total=1000, clamp=None):
    if ctrl_type == "static":
        w = w0
    elif ctrl_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    elif ctrl_type == "cosine":
        w = w0 * (torch.cos(torch.pi * t / t_total) + 1)
    elif ctrl_type == "inv_linear":
        w = w0 * 2 * (t / t_total)
    elif ctrl_type == "sine":
        w = w0 * (torch.sin(torch.pi * t / t_total - torch.pi / 2) + 1)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w


def aggregate_noise_pred(
    noise_pred_uncond,
    noise_pred_cond,
    t,
    w_src,
    w_tgt,
    w_src_ctrl_type,
    w_tgt_ctrl_type,
    ctrl_mode,
    removal_version=2,
):
    delta_noise_pred_src = noise_pred_cond[0] - noise_pred_uncond[0]
    delta_noise_pred_tgt = noise_pred_cond[1] - noise_pred_uncond[1]

    w_src_cur = ctrl_weight(t, w_src, w_src_ctrl_type)
    w_tgt_cur = ctrl_weight(t, w_tgt, w_tgt_ctrl_type)

    if ctrl_mode == "add":
        aggregated_noise = add_aggregator_v1(
            delta_noise_pred_src,
            w_src_cur,
            delta_noise_pred_tgt,
            w_tgt_cur,
            mode="latent",
        )
    elif ctrl_mode == "remove":
        remove_aggregator = (
            remove_aggregator_v1 if removal_version == 1 else remove_aggregator_v2
        )
        aggregated_noise = remove_aggregator(
            delta_noise_pred_src,
            w_src_cur,
            delta_noise_pred_tgt,
            w_tgt_cur,
            mode="latent",
        )
    else:
        raise ValueError("Unrecognized prompt ctrl mode")

    return aggregated_noise

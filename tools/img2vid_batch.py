import os
import imageio
import numpy as np
import torch
import torchvision
from torchvision.io import read_image
from einops import rearrange


save_as_gif = False
out_dir = 'temp/videos'
paths = [
    # Original
    'metrics/videos/src/add/car_turn',
    'metrics/videos/src/add/digital_woman_butterfly',
    'metrics/videos/src/add/sakura_matou_hat',
    'metrics/videos/src/rm/panda_guitar',
    'metrics/videos/src/rm/puppy_dandelions',
    'metrics/videos/src/rm/woman_car_sunglasses',
    'metrics/videos/src/style/bird_pop_art',
    'metrics/videos/src/style/car_street_cyberpunk',
    'metrics/videos/src/style/gwen_book',
    # AnimateDiff
    'metrics/videos/tgt/animatediff/add/car_one_helicopter_1.0_4.5',
    'metrics/videos/tgt/animatediff/add/digital_woman_butterfly_1.0_0.6',
    'metrics/videos/tgt/animatediff/add/sakura_matou_hat_1.0_0.6',
    'metrics/videos/tgt/animatediff/rm/panda_guitar_0.9_0.6',
    'metrics/videos/tgt/animatediff/rm/puppy_dandelions_1.0_0.5',
    'metrics/videos/tgt/animatediff/rm/woman_car_sunglasses_1.0_0.1',
    'metrics/videos/tgt/animatediff/style/bird_pop_art_1.0_5.4',
    'metrics/videos/tgt/animatediff/style/car_street_cyberpunk_1.0_5.25',
    'metrics/videos/tgt/animatediff/style/gwen_book_cyberpunk_1.0_3.0',
    # FateZero
    'metrics/videos/tgt/fatezero/add/car_one_helicopter_fatezero',
    'metrics/videos/tgt/fatezero/add/digital_woman_butterfly_fatezero',
    'metrics/videos/tgt/fatezero/add/sakura_matou_hat_fatezero',
    'metrics/videos/tgt/fatezero/rm/panda_guitar_fatezero',
    'metrics/videos/tgt/fatezero/rm/puppy_dandelions_fatezero',
    'metrics/videos/tgt/fatezero/rm/woman_car_sunglasses_fatezero',
    'metrics/videos/tgt/fatezero/style/bird_pop_art_fatezero',
    'metrics/videos/tgt/fatezero/style/car_street_cyberpunk_fatezero',
    'metrics/videos/tgt/fatezero/style/gwen_cyberpunk_fatezero',
    # FLATTEN
    'metrics/videos/tgt/flatten/add/car_one_helicopter_flatten',
    'metrics/videos/tgt/flatten/add/digital_woman_butterfly_flatten',
    'metrics/videos/tgt/flatten/add/sakura_matou_hat_flatten',
    'metrics/videos/tgt/flatten/rm/panda_guitar_flatten',
    'metrics/videos/tgt/flatten/rm/puppy_dandelions_flatten',
    'metrics/videos/tgt/flatten/rm/woman_car_sunglasses_flatten',
    'metrics/videos/tgt/flatten/style/bird_pop_art_flatten',
    'metrics/videos/tgt/flatten/style/car_street_cyberpunk_flatten',
    'metrics/videos/tgt/flatten/style/gwen_cyberpunk_flatten',
    # Rerender-A-Video
    'metrics/videos/tgt/rav/add/car_one_helicopter_rav',
    'metrics/videos/tgt/rav/add/digital_woman_butterfly_rav',
    'metrics/videos/tgt/rav/add/sakura_matou_hat_rav',
    'metrics/videos/tgt/rav/rm/panda_guitar_rav',
    'metrics/videos/tgt/rav/rm/puppy_dandelions_rav',
    'metrics/videos/tgt/rav/rm/woman_car_sunglasses_rav',
    'metrics/videos/tgt/rav/style/bird_pop_art_rav',
    'metrics/videos/tgt/rav/style/car_street_cyberpunk_rav',
    'metrics/videos/tgt/rav/style/gwen_cyberpunk_rav',
    # TokenFlow
    'metrics/videos/tgt/tokenflow/add/car_one_helicopter_tokenflow',
    'metrics/videos/tgt/tokenflow/add/digital_woman_butterfly_tokenflow',
    'metrics/videos/tgt/tokenflow/add/sakura_matou_hat_tokenflow',
    'metrics/videos/tgt/tokenflow/rm/panda_guitar_tokenflow',
    'metrics/videos/tgt/tokenflow/rm/puppy_dandelions_tokenflow',
    'metrics/videos/tgt/tokenflow/rm/woman_car_sunglasses_tokenflow',
    'metrics/videos/tgt/tokenflow/style/bird_pop_art_tokenflow',
    'metrics/videos/tgt/tokenflow/style/car_street_cyberpunk_tokenflow',
    'metrics/videos/tgt/tokenflow/style/gwen_cyberpunk_tokenflow',
    # Vid2Vid-Zero
    'metrics/videos/tgt/v2v_zero/add/car_one_helicopter_v2v_zero',
    'metrics/videos/tgt/v2v_zero/add/digital_woman_butterfly_v2v_zero',
    'metrics/videos/tgt/v2v_zero/add/sakura_matou_hat_v2v_zero',
    'metrics/videos/tgt/v2v_zero/rm/panda_guitar_v2v_zero',
    'metrics/videos/tgt/v2v_zero/rm/puppy_dandelions_v2v_zero',
    'metrics/videos/tgt/v2v_zero/rm/woman_car_sunglasses_v2v_zero',
    'metrics/videos/tgt/v2v_zero/style/bird_pop_art_v2v_zero',
    'metrics/videos/tgt/v2v_zero/style/car_street_cyberpunk_v2v_zero',
    'metrics/videos/tgt/v2v_zero/style/gwen_cyberpunk_v2v_zero',
    # VidToMe
    'metrics/videos/tgt/vidtome/add/car_one_helicopter_vidtome',
    'metrics/videos/tgt/vidtome/add/digital_woman_butterfly_vidtome',
    'metrics/videos/tgt/vidtome/add/sakura_matou_hat_vidtome',
    'metrics/videos/tgt/vidtome/rm/panda_guitar_vidtome',
    'metrics/videos/tgt/vidtome/rm/puppy_dandelions_vidtome',
    'metrics/videos/tgt/vidtome/rm/woman_car_sunglasses_vidtome',
    'metrics/videos/tgt/vidtome/style/bird_pop_art_vidtome',
    'metrics/videos/tgt/vidtome/style/car_street_cyberpunk_vidtome',
    'metrics/videos/tgt/vidtome/style/gwen_cyberpunk_vidtome',
]


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # t: n_frames; b: n_batches, if only 1 video then 1
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # outputs is a list of arrays each of shape (H, W, C)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        imageio.mimsave(path, outputs, fps=fps)
    except:
        imageio.mimsave(path, outputs, duration=1000 * 1 / fps)


if __name__ == "__main__":
    postfix = '.gif' if save_as_gif else '.mp4'

    for img_dir in paths:
        images = [img for img in sorted(os.listdir(img_dir)) if img.endswith(".png")]
        video = torch.stack(
            [read_image(os.path.join(img_dir, img)) for img in images], dim=0
        )  # (F, C, H, W)
        video = video / 127.5 - 1.0  # normalize to [-1, 1]
        videos = rearrange(video, "(b f) c h w -> b c f h w", b=1)

        baseline_name = img_dir.split('/')[-3]
        save_videos_grid(
            videos,
            os.path.join(out_dir, baseline_name, f"{os.path.basename(img_dir)}{postfix}"),
            rescale=True,
            fps=8,
        )

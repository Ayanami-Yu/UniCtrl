import os
import subprocess


modes = ["add", "rm", "style"]
models = [
    "animatediff",
    "fatezero",
    "flatten",
    "rav",
    "tokenflow",
    "v2v_zero",
    "vidtome",
]
vid_dirs = [f"metrics/videos/src/{mode}" for mode in modes] + [
    f"metrics/videos/tgt/{model}/{mode}" for model in models for mode in modes
]
out_dirs = [f"temp/videos/src/{mode}" for mode in modes] + [
    f"temp/videos/tgt/{model}/{mode}" for model in models for mode in modes
]

for i in range(len(vid_dirs)):
    for name in os.listdir(vid_dirs[i]):
        cur_dir = os.path.join(vid_dirs[i], name)
        os.makedirs(out_dirs[i], exist_ok=True)

        subprocess.run(
            [
                "python",
                "tools/img2vid.py",
                "--image_dir",
                cur_dir,
                "--output_dir",
                out_dirs[i],
                "--save_as_gif"
            ],
            capture_output=False,
            text=True,
        )

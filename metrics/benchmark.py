import subprocess
import re
import yaml


output_path = "metrics/benchmark.yaml"

modalities = ["image", "video"]
modes = ["add", "rm", "style"]
baselines = {
    "image": ["cg", "ledits_pp", "masactrl", "mdp", "p2p", "sd", "sega"],
    "video": [
        "animatediff",
        "fatezero",
        "flatten",
        "rav",
        "tokenflow",
        "v2v_zero",
        "vidtome",
    ],
}
configs = {
    "image": "metrics/images_v2.yaml",
    "video": "metrics/videos_v2.yaml",
}
data = {
    "clip_score": {
        "image": {
            "add": {},
            "rm": {},
            "style": {},
        },
        "video": {
            "add": {},
            "rm": {},
            "style": {},
        },
    },
    "clip_dir": {
        "image": {
            "add": {},
            "rm": {},
            "style": {},
        },
        "video": {
            "add": {},
            "rm": {},
            "style": {},
        },
    },
}

for modality in modalities:
    for mode in modes:
        for model in baselines[modality]:
            # compute CLIP similarity
            res_score = subprocess.run(
                [
                    f"python",
                    "metrics/clip_score.py",
                    "--modality",
                    modality,
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--config_file",
                    configs[modality],
                ],
                capture_output=True,
                text=True,
            )

            # extract float using regex
            match = re.search(r"[-+]?\d*\.\d+", res_score.stdout)
            if match:
                data["clip_score"][modality][mode][model] = float(match.group())
                print(f'CLIP score for {mode} of {model} is {float(match.group())}')
            else:
                raise RuntimeError("No float value found in output")

            # compute CLIP directional similarity
            res_dir = subprocess.run(
                [
                    f"python",
                    "metrics/directional_clip.py",
                    "--modality",
                    modality,
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--config_file",
                    configs[modality],
                ],
                capture_output=True,
                text=True,
            )
            match = re.search(r"[-+]?\d*\.\d+", res_dir.stdout)
            if match:
                data["clip_dir"][modality][mode][model] = float(match.group())
                print(f'CLIP dir for {mode} of {model} is {float(match.group())}')
            else:
                raise RuntimeError("No float value found in output")

with open(output_path, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

import os
import re
import yaml
from pathlib import Path
from PIL import Image


res_path = "exp/animatediff/results_selected"
output_file = "videos_v2.yaml"
prev_file = "metrics/videos_v1.yaml"

content = {
    "add": {},
    "rm": {},
    "style": {},
}

# load the previous yaml file if provided
if prev_file:
    with open(prev_file, "r") as f:
        data_prev = yaml.safe_load(f)
    for mode in data_prev.keys():
        content[mode] = {k: v for k, v in data_prev[mode].items()}

for vid_name in os.listdir(res_path):
    vid_dir = os.path.join(res_path, vid_name)
    if not os.path.isdir(vid_dir):
        continue

    with open(os.path.join(vid_dir, "configs.yaml"), "r") as config_file:
        conf = yaml.safe_load(config_file)

    # first load the images
    dir_path = Path(vid_dir)
    fields = ["src", "tgt"]
    vid_paths = [next(Path(vid_dir).glob(f"*{fields[i]}*"), None) for i in range(2)]
    if vid_paths[0] and vid_paths[1]:
        vids = [[Image.open(os.path.join(vid_paths[i], img)) for img in sorted(os.listdir(vid_paths[i])) if img.endswith(".png")] for i in range(2)]
    else:
        raise FileNotFoundError()

    # regular expression to capture numbers after 'src_' or 'tgt_'
    pattern = re.compile(r"_(?:src|tgt)_(\d+\.\d+_-?\d+\.\d+)$")
    extracted_tgt = match.group(1) if (match := pattern.search(vid_paths[1].name)) else None

    # save the images to target directory
    vid_save_paths = [
        f"metrics/videos/src/{conf['ctrl_mode']}/{vid_name}",
        f"metrics/videos/tgt/animatediff/{conf['ctrl_mode']}/{vid_name}_{extracted_tgt}",
    ]
    for i in range(len(vids)):
        os.makedirs(vid_save_paths[i], exist_ok=True)
        for j in range(len(vids[i])):
            vids[i][j].save(f"{vid_save_paths[i]}/%05d.png" % j)

    # add a new case into content
    content[conf["ctrl_mode"]][vid_name] = {
        "seed": conf["seed"],
        "src_prompt": conf["prompts"][0],
        "tgt_prompt": {
            "default": conf["tgt_prompt_default"],
            "change": conf["prompts"][1],
        } if conf["ctrl_mode"] == "rm" else conf["prompts"][1],
        "src_images": vid_save_paths[0],
        "tgt_images": {
            "animatediff": vid_save_paths[1],
            "fatezero": "",
            "tokenflow": "",
            "vidtome": "",
            "flatten": "",
            "v2v_zero": "",
            "rav": "",
        },
    }

with open(output_file, "w") as f:
    yaml.dump(content, f, default_flow_style=False)

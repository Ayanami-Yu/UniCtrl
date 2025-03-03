import os
import re
import yaml
from pathlib import Path
from PIL import Image


res_path = "exp/sd/pie_results_selected"
output_file = "images_v2.yaml"
prev_file = "images_v1.yaml"

content = {
    "add": {},
    "rm": {},
    "style": {},
}

# load the previous yaml file if provided
if prev_file:
    with open(prev_file, 'r') as f:
        data_prev = yaml.safe_load(f)
    for mode in data_prev.keys():
        content[mode] = {k: v for k, v in data_prev[mode].items()}

for img_name in os.listdir(res_path):
    img_dir = os.path.join(res_path, img_name)
    if not os.path.isdir(img_dir):
        continue

    with open(os.path.join(img_dir, "configs.yaml"), "r") as config_file:
        conf = yaml.safe_load(config_file)

    # first load the images
    dir_path = Path(img_dir)
    fields = ["src", "tgt"]
    img_paths = [next(Path(img_dir).glob(f"*{fields[i]}*.png"), None) for i in range(2)]
    if img_paths[0] and img_paths[1]:
        imgs = [Image.open(img_paths[i]) for i in range(2)]
    else:
        raise FileNotFoundError()

    # regular expression to capture numbers after 'src_' or 'tgt_'
    pattern = re.compile(r"_(?:src|tgt)_(\d+\.\d+_\d+\.\d+)\.png$")
    extracted = [
        match.group(1) if (match := pattern.search(img_paths[i].name)) else None
        for i in range(2)
    ]

    # save the images to target directory
    mid_dirs = ["default", "sd"]
    img_save_paths = [
        f"metrics/images/{fields[i]}/{mid_dirs[i]}/{conf['ctrl_mode']}/{img_name}_{extracted[i]}.png"
        for i in range(2)
    ]
    for i in range(2):
        imgs[i].save(img_save_paths[i])

    # add a new case into content
    content[conf['ctrl_mode']][img_name] = {
        "seed": conf['seed'],
        "src_prompt": conf['prompts'][0],
        "tgt_prompt": (
            {
                "default": "",
                "sd": conf['prompts'][1],
            }
            if conf['ctrl_mode'] == "rm"
            else conf['prompts'][1]
        ),
        "src_image": {
            "default": img_save_paths[0],
            "sega": "",
        },
        "tgt_image": {
            "sd": img_save_paths[1],
            "masactrl": "",
            "p2p": "",
            "sega": "",
            "ledits_pp": "",
            "mdp": "",
            "cg": "",
        },
    }

with open(output_file, "w") as f:
    yaml.dump(content, f, default_flow_style=False)

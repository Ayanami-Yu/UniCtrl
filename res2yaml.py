import os
import yaml


class ConfigAdd:
    def __init__(self, seed, src_prompt, tgt_prompt, src_img, tgt_img):
        self.seed = seed
        self.src_prompt = src_prompt
        self.tgt_prompt = tgt_prompt
        self.src_image = {
            'default': src_img,
            'sega': '',
        }
        self.tgt_image = {
            'sd': tgt_img,
            'masactrl': '',
            'p2p': '',
            'sega': '',
            'ledits_pp': '',
            'mdp': '',
            'cg': '',
        }

class ConfigRm:
    def __init__(self, seed, src_prompt, tgt_prompt, src_img, tgt_img):
        self.seed = seed
        self.src_prompt = src_prompt
        self.tgt_prompt = {

        }
        self.src_image = {
            'default': src_img,
            'sega': '',
        }
        self.tgt_image = {
            'sd': tgt_img,
            'masactrl': '',
            'p2p': '',
            'sega': '',
            'ledits_pp': '',
            'mdp': '',
            'cg': '',
        }


res_path = 'exp/sd/pie_results_selected'
output_file = 'image_configs.yaml'

content = {
    'add': [],
    'rm': [],
    'style': [],
}

with open(output_file, 'w') as f:
    for img_dir in os.listdir(res_path):
        img_dir = os.path.join(res_path, img_dir)
        if not os.path.isdir(img_dir):
            continue

        with open(os.path.join(img_dir, 'configs.yaml'), 'r') as config_file:
            data = yaml.safe_load(config_file)
        
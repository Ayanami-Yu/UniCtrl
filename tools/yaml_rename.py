import os
import yaml


modality = 'images'  # images or videos
config_file = "metrics/images_v2.yaml" if modality == 'images' else 'metrics/videos_v2.yaml'
method_exception = 'sd' if modality == 'images' else 'animatediff'
sub_dir = 'tgt_image' if modality == 'images' else 'tgt_images'
postfix = '.png' if modality == 'images' else ''

prefix = '/home/hongyu/UniCtrl/'

with open(config_file, "r") as f:
    data = yaml.safe_load(f)

for mode in data.keys():
    for name in data[mode].keys():
        for method in data[mode][name][sub_dir].keys():
            if method == method_exception:
                continue

            path_old = data[mode][name][sub_dir][method]
            path_new = f'metrics/{modality}/tgt/{method}/{mode}/{name}_{method}' + postfix
            if os.path.exists(prefix + path_new):
                continue

            try:  # FIXME temp
                os.rename(prefix + path_old, prefix + path_new)
            except:
                if method == 'sega':
                    path_old = f'metrics/{modality}/tgt/{method}/{mode}/{name}_{method}_tgt.png'
                elif method == 'masactrl':
                    path_old = f'metrics/{modality}/tgt/{method}/{mode}/{name}_{method}_step4_layer10.png'
                os.rename(prefix + path_old, prefix + path_new)
            
            data[mode][name][sub_dir][method] = path_new

            print(f'path renamed from {path_old} to {path_new}')

with open(config_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

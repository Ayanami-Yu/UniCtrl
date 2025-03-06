import yaml


config_file = "metrics/images_v2.yaml"

with open(config_file, "r") as f:
    data = yaml.safe_load(f)

for mode in data.keys():
    for name in data[mode].keys():
        if mode == "rm":
            data[mode][name]["tgt_prompt"]["change"] = data[mode][name][
                "tgt_prompt"
            ].pop("sd")
        else:
            tgt_prompt_default = data[mode][name]["tgt_prompt"]
            data[mode][name]["tgt_prompt"] = {
                "default": tgt_prompt_default,
                "change": "",
            }

with open(config_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

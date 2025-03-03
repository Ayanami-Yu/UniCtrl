import os
import yaml
import ast  # To safely evaluate Python literals


res_path = "exp/sd/pie_results_selected"

for img_dir in os.listdir(res_path):
    img_dir = os.path.join(res_path, img_dir)
    if not os.path.isdir(img_dir):
        continue

    txt_file = os.path.join(img_dir, "configs.txt")
    yaml_file = os.path.join(img_dir, "configs.yaml")

    data = {}

    with open(txt_file, "r") as f:
        for line in f:
            key, value = line.strip().split(": ", 1)  # Split only on the first occurrence

            # Convert values appropriately
            if value.startswith("[") or value.startswith("("):  # Convert lists/tuples
                value = ast.literal_eval(value)
            elif value.isdigit():  # Convert integers
                value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:  # Convert floats
                value = float(value)
            elif value.startswith("'") or value.startswith('"'):  # Convert strings
                value = value.strip("'\"")

            data[key] = value

    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Converted YAML saved to {yaml_file}")
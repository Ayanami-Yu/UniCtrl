import re
import json
import yaml

from collections import namedtuple


json_path = "data/mapping_file.json"
yaml_path = "pie_prompts.yaml"
edit_category_dict = {
    "2": "add",
    "3": "rm",
    "9": "style",
}
content = {
    "add": [],
    "rm": [],
    "style": [],
}
Case = namedtuple("Case", ["src_prompt", "tgt_prompt"])
RmTarget = namedtuple("RmTarget", ["default", "sd"])

with open(json_path, "r") as f:
    edit_instructions = json.load(f)

for key, item in edit_instructions.items():
    if item["editing_type_id"] not in edit_category_dict.keys():
        continue

    edit_type = edit_category_dict[item["editing_type_id"]]
    original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
    editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

    if edit_type == "rm":
        # Use regex to extract the content inside square brackets
        match = re.search(r"\[(.*?)\]", item["original_prompt"])
        prompt_sd = match.group(1) if match else ""
        content[edit_type].append(
            Case(
                src_prompt=original_prompt,
                tgt_prompt=RmTarget(default=editing_prompt, sd=prompt_sd)._asdict(),
            )._asdict()
        )
    else:
        content[edit_type].append(
            Case(src_prompt=original_prompt, tgt_prompt=editing_prompt)._asdict()
        )

# Convert list to dict and assign case numbers
for key in content.keys():
    content[key] = {i: case for i, case in enumerate(content[key])}

with open(yaml_path, "w") as f:
    yaml.dump(content, f)

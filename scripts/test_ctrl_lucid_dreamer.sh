#!/bin/bash
# usage: bash scripts/test_ctrl_lucid_dreamer.sh

config_name="lion_water"
workspace="${config_name}"
devices=(7)
w_src=1.0
w_tgt=(1.00)

# ctrl_mode="remove"
ctrl_mode="add"
removal_version=2

for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES=${devices[$i]} nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src} --w_tgt_cli=${w_tgt[$i]} --workspace_cli="${workspace}/${w_src}_${w_tgt[$i]}" --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} > nohup/${config_name}.txt 2>&1 &
done
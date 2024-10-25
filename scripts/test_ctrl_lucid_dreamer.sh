#!/bin/bash
# usage: bash scripts/test_ctrl_lucid_dreamer.sh

config_name="cat_armor"
workspace="${config_name}_rm_v2"
devices=(1 2 3 4 5 6)
w_src=1.0
w_tgt=(0.6 0.8 1.0 -1.0 1.2 1.4)

ctrl_mode="remove"
removal_version=2

for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES=${devices[$i]} nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src} --w_tgt_cli=${w_tgt[$i]} --workspace_cli="${workspace}/${w_src}_${w_tgt[$i]}" --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} > nohup/${config_name}.txt 2>&1 &
done
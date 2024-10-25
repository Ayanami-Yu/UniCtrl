#!/bin/bash

config_name="pikachu_crown"
devices=(1 2 3 4 5 6)
w_src=1.0
w_tgt=(-1.00 -0.85 -0.70 -0.55 -0.40 -0.25)

ctrl_mode="remove"
removal_version=2

for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES=${devices[$i]} nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src} --w_tgt_cli=${w_tgt[$i]} --workspace_cli="${config_name}/${w_src}_${w_tgt[$i]}" --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} > nohup/${config_name}.txt 2>&1 &
done
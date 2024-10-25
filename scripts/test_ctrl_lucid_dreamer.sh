#!/bin/bash

config_name="pikachu_crown"
devices=(1 2 3 4 5 6)
w_src=1.0
w_tgt_start=-1.0
w_tgt_inc=0.18
ctrl_mode="remove"
removal_version=2

n_tests=${#devices[@]}

for(( i = 0; i < ${n_tests}; i++)); do
    w_tgt=`awk -v x=${w_tgt_start} -v y=${i} -v z=${w_tgt_inc} 'BEGIN {printf "%.2f", x + y * z}'`
    
    echo "CUDA_VISIBLE_DEVICES="${devices[$i]}" nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src} --w_tgt_cli=${w_tgt} --workspace_cli='${config_name}/${w_src}_${w_tgt}' --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} &"

    CUDA_VISIBLE_DEVICES="${devices[$i]}" nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src} --w_tgt_cli=${w_tgt} --workspace_cli='${config_name}/${w_src}_${w_tgt}' --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} &
done

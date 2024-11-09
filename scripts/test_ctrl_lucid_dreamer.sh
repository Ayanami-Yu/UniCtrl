#!/bin/bash
# Usage: bash scripts/test_ctrl_lucid_dreamer.sh

config_name="fox_rock"
devices=(1 2 3 4 5 6 7)
w_src=1.0
w_tgt=(0.05 0.2 0.4 0.8 1.2 1.6 1.8)

ctrl_mode="add"  # add or remove (rm)
w_tgt_ctrl_type="static"
removal_version=2

# w_src should be a list if not fixed
fix_w_src="true"
workspace="${config_name}_${ctrl_mode}_${w_tgt_ctrl_type}"

if [ "${ctrl_mode}" == "rm" ]; then
    ctrl_mode="remove"
fi

for i in "${!devices[@]}"; do
    if [ "$fix_w_src" == "true" ]; then
        w_src_cur=${w_src}
    else
        w_src_cur=${w_src[$i]}
    fi
    
    CUDA_VISIBLE_DEVICES=${devices[$i]} nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/${config_name}.yaml --w_src_cli=${w_src_cur} --w_tgt_cli=${w_tgt[$i]} --workspace_cli="${workspace}/${w_src_cur}_${w_tgt[$i]}" --ctrl_mode_cli ${ctrl_mode} --removal_version_cli=${removal_version} --w_tgt_ctrl_type ${w_tgt_ctrl_type} > nohup/${config_name}.txt 2>&1 &
done
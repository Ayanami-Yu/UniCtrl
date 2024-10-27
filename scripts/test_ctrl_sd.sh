#!/bin/bash
# Usage: bash scripts/test_ctrl_sd.sh

device=3
src_prompt='"a Ragdoll cat wearing armor, full body shot"'
tgt_prompt='"ar armor"'

src_params=(0.8 0.1 7)
tgt_params=(-1.0 0.1 30)
ctrl_mode="remove"
removal_version=2
w_tgt_ctrl_type="static"
name="ragdoll_armor"

if [ "${ctrl_mode}" == "remove" ]; then
    workspace="${name}_rm_v${removal_version}_${w_tgt_ctrl_type}"
else
    workspace="${name}_add_${w_tgt_ctrl_type}"
fi

CUDA_VISIBLE_DEVICES=${device} nohup python scripts/test_ctrl_sd.py --prompt "${src_prompt}" "${tgt_prompt}" --out_dir "./exp/sd/${workspace}/" --src_params ${src_params[@]} --tgt_params ${tgt_params[@]} --ctrl_mode "${ctrl_mode}" --removal_version ${removal_version} --w_tgt_ctrl_type "${w_tgt_ctrl_type}" > nohup/${name} 2>&1 &
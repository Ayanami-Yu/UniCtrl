#!/bin/bash
# Usage: bash scripts/test_ctrl.sh

device=1
src_prompt='A porsche car turning a corner in the countryside.'
tgt_prompt='A porsche car turning a corner in the countryside followed by helicopters.'

name='temp'
ctrl_mode='add'  # add or remove (rm)
model='ad'  # sd or animatediff (ad)
src_params=(1.0 0.1 1)
w_tgt_ctrl_type='cosine'
removal_version=2

if [ "${ctrl_mode}" == "rm" ]; then
    tgt_params=(-1.0 0.1 36)
    workspace="${name}_rm_v${removal_version}_${w_tgt_ctrl_type}"
else
    tgt_params=(0.0 0.3 1)  # add
    workspace="${name}_add_${w_tgt_ctrl_type}"
fi

if [ "${model}" == "ad" ]; then
    model="animatediff"
fi

if [ "${ctrl_mode}" == "rm" ]; then
    ctrl_mode="remove"
fi

CUDA_VISIBLE_DEVICES=${device} nohup python scripts/test_ctrl_${model}.py --prompt "${src_prompt}" "${tgt_prompt}" --out_dir "./exp/${model}/${workspace}/" --src_params ${src_params[@]} --tgt_params ${tgt_params[@]} --ctrl_mode "${ctrl_mode}" --removal_version ${removal_version} --w_tgt_ctrl_type "${w_tgt_ctrl_type}" > "nohup/${name}.txt" 2>&1 &
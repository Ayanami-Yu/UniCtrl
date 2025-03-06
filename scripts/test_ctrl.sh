#!/bin/bash
# Usage: bash scripts/test_ctrl.sh

device=6
src_prompt='a digital art of a brown hair woman'
tgt_prompt='a digital art of a brown hair woman with flying butterfly'

name='ablation'
ctrl_mode='add'  # add or rm (remove)
model='sd'  # sd, ad (animatediff), or lgm
src_params=(1.0 0.1 1)
w_tgt_ctrl_type='cosine'
removal_version=2
seed=644051455

if [ "${ctrl_mode}" == "rm" ]; then
    tgt_params=(-1.0 0.1 5)
    workspace="${name}_rm_v${removal_version}_${w_tgt_ctrl_type}"
else
    tgt_params=(0.0 0.1 10)  # add
    workspace="${name}_add_${w_tgt_ctrl_type}"
fi

if [ "${model}" == "ad" ]; then
    model="animatediff"
fi

if [ "${ctrl_mode}" == "rm" ]; then
    ctrl_mode="remove"
fi

if [ "${model}" == "lgm" ]; then
    CUDA_VISIBLE_DEVICES=${device} nohup python scripts/test_ctrl_${model}.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/${model}/${workspace}/ --prompts "${src_prompt}" "${tgt_prompt}" --src_params ${src_params[@]} --tgt_params ${tgt_params[@]} --ctrl_mode "${ctrl_mode}" --removal_version ${removal_version} --w_tgt_ctrl_type "${w_tgt_ctrl_type}" > "nohup/${name}.txt" 2>&1 &

else
    CUDA_VISIBLE_DEVICES=${device} nohup python scripts/test_ctrl_${model}.py --prompt "${src_prompt}" "${tgt_prompt}" --seed ${seed} --out_dir "./exp/${model}/${workspace}/" --src_params ${src_params[@]} --tgt_params ${tgt_params[@]} --ctrl_mode "${ctrl_mode}" --removal_version ${removal_version} --w_tgt_ctrl_type "${w_tgt_ctrl_type}" > "nohup/${name}.txt" 2>&1 &
fi
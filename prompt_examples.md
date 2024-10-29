# Prompts Examples

All use `add_aggregator_v1`

## Image

### Examples

#### Add

[
    "Catwoman holding a sniper rifle",
    "Catwoman holding a sniper rifle and wearing a hat",
]

src = 1.3 ~ 1.4
tgt = 1.0 ~ 3.0

[
    "an astronaut riding a horse",
    "an astronaut riding a horse and holding a Gatling gun",
]

src = 0.9
tgt = 0.5 ~ 1.5

#### Remove

[
    "a Ragdoll cat wearing armor, full body shot",
    "ar armor",
]

[
    "jack o lantern hanging on a tree",
    "jack o lantern",
]

[
    "Iron Man is walking towards the camera in the rain at night, with a lot of fog behind him. Science fiction movie, close-up",
    "rain, a lot of fog",
]

### CLI

#### Add

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_sd.py --prompt "an astronaut is riding a horse on a road" "an astronaut is riding a horse and holding a Gatling gun on a road" --out_dir "./exp/sd/astronaut_horse_gun_road/" --src_params 0.7 0.1 6 --tgt_params 0.0 0.1 25 --ctrl_mode "add" --w_tgt_ctrl_type "linear" > nohup/astronaut_horse_gun_road 2>&1 &`

`CUDA_VISIBLE_DEVICES=2 nohup python scripts/test_ctrl_sd.py --prompt "Catwoman holding a sniper rifle" "Catwoman holding a sniper rifle and wearing a hat" --out_dir "./exp/sd/catwoman_rifle_linear/" --src_params 0.9 0.1 7 --tgt_params 0.0 0.1 30 --ctrl_mode "add" --w_tgt_ctrl_type "linear" > nohup/catwoman_rifle 2>&1 &`

`CUDA_VISIBLE_DEVICES=2 nohup python scripts/test_ctrl_sd.py --prompt "Catwoman holding a sniper rifle" "Catwoman holding a sniper rifle and wearing a hat" --out_dir "./exp/sd/catwoman_rifle_cosine/" --src_params 0.9 0.1 7 --tgt_params 0.0 0.1 30 --ctrl_mode "add" --w_tgt_ctrl_type "cosine" > nohup/catwoman_rifle 2>&1 &`

#### Remove

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_sd.py --prompt "Catwoman holding a sniper rifle" "a sniper rifle" --out_dir "./exp/sd/catwoman_rifle_rm/" --src_params 2.3 0.1 2 --tgt_params 0.0 0.1 23 --ctrl_mode "remove" > nohup/catwoman_rifle_rm 2>&1 &`

`CUDA_VISIBLE_DEVICES=0 nohup python scripts/test_ctrl_sd.py --prompt "Catwoman holding a sniper rifle" "a sniper rifle" --out_dir "./exp/sd/catwoman_rifle_rm_v2/" --src_params 0.9 0.1 3 --tgt_params -1.0 0.1 23 --ctrl_mode "remove" --removal_version 2 > nohup/catwoman_rifle_rm 2>&1 &`

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_sd.py --prompt "Catwoman holding a sniper rifle" "a sniper rifle" --out_dir "./exp/sd/catwoman_rifle_rm_fix_scale/" --scale 1.0 --theta_params 0.0 0.025 62 --ctrl_mode "remove" > nohup/catwoman_rifle_rm 2>&1 &`

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_sd.py --prompt "a DSLR photo of a cat wearing armor" "an armor" --out_dir "./exp/sd/cat_armor/" --scale 1.0 --theta_params 0.0 0.1 62 --ctrl_mode "remove" > nohup/cat_armor 2>&1 &`

`CUDA_VISIBLE_DEVICES=3 nohup python scripts/test_ctrl_sd.py --prompt "an astronaut riding a horse" "an astronaut" --out_dir "./exp/sd/astronaut_horse_rm_v2_inv_linear/" --src_params 0.9 0.1 7 --tgt_params -1.0 0.1 30 --ctrl_mode "remove" --removal_version 2 > nohup/astronaut_horse 2>&1 &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_sd.py --prompt "a Ragdoll cat wearing armor" "an armor" --out_dir "./exp/sd/ragdoll_armor/" --src_params 0.8 0.1 7 --tgt_params -1.0 0.1 30 --ctrl_mode "remove" --removal_version 2 --w_tgt_ctrl_type "inv_linear" > nohup/ragdoll_armor 2>&1 &`

## Video

### Examples

[
    "a silver wolf is running",
    "a silver wolf is running after a golden eagle",
]

[
    "an astronaut is riding a horse on a road",
    "an astronaut is riding a horse and holding a Gatling gun on a road",
]

[
    "an astronaut is riding a horse",
    "an astronaut holding a Gatling gun is riding a horse",
]

[
    "Catwoman is holding a sniper rifle",
    "Catwoman holding a sniper rifle is wearing a hat on her head",
]

[
    "a panda is playing guitar on times square",
    "a panda is playing guitar on times square, with a drum next to it",
]

[
    "a high quality realistic photo of a cute cat running in a beautiful meadow",
    "a high quality realistic photo of a cute cat with large wings running in a beautiful meadow",
]

[
    "a woman is walking in the rain",
    "a woman is walking in the rain and carrying a red handbag",
]

src_prompt='"A gentleman with a handlebar mustache, a bowler hat, and a monocle"'
tgt_prompt='"a handlebar mustache, a monocle"'

### CLI

#### Text2Video-Zero

`CUDA_VISIBLE_DEVICES=3 nohup python scripts/test_ctrl_video_zero.py --prompt "an astronaut is riding a horse" "an astronaut holding a Gatling gun is riding a horse" --out_dir "./exp/video_zero/astronaut_horse_gun/" --src_params 0.9 0.1 1 --tgt_params 0.0 0.02 80 > nohup/astronaut_horse_gun.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=4 nohup python scripts/test_ctrl_video_zero.py --prompt "a woman is walking in the rain" "a woman is walking in the rain and carrying a red handbag" --out_dir "./exp/video_zero/woman_rain_handbag/" --src_params 0.9 0.1 1 --tgt_params 0.0 0.02 80 > nohup/woman_rain_handbag.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=5 nohup python scripts/test_ctrl_video_zero.py --prompt "a horse galloping on the street, best quality" "a horse galloping on the street with a girl riding on it, best quality" --out_dir "./exp/video_zero/horse_girl/" --src_params 0.9 0.1 1 --tgt_params 0.0 0.02 80 > nohup/horse_girl.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=6 nohup python scripts/test_ctrl_video_zero.py --prompt "a high quality realistic photo of a cute cat running in a beautiful meadow" "a high quality realistic photo of a cute cat running in a beautiful meadow, Van Gogh style" --out_dir "./exp/video_zero/cat_meadow_van_gogh/" --src_params 0.9 0.1 1 --tgt_params 0.0 0.2 80 > nohup/cat_meadow_van_gogh.txt 2>&1 &`

#### AnimateDiff

##### Add

###### Good

`nohup python scripts/test_ctrl_animatediff.py --prompt "an astronaut is riding a horse" "an astronaut holding a Gatling gun is riding a horse" --out_dir "./exp/animatediff/astronaut_horse_gun/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 --gpu 3 > nohup/astronaut_horse_gun.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=4 nohup python scripts/test_ctrl_animatediff.py --prompt "a panda is playing guitar on times square" "a panda is playing guitar on times square, with a drum next to it" --out_dir "./exp/animatediff/panda_guitar_drum/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/panda_guitar_drum.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=3 nohup python scripts/test_ctrl_animatediff.py --prompt "a high quality realistic photo of a cute cat running in a beautiful meadow" "a high quality realistic photo of a cute cat with large wings running in a beautiful meadow" --out_dir "./exp/animatediff/cat_meadow_wings/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/cat_meadow_wings.txt 2>&1 &`

`nohup python scripts/test_ctrl_animatediff.py --prompt "a woman is walking in the rain" "a woman is walking in the rain and carrying a red handbag" --out_dir "./exp/animatediff/woman_rain_handbag/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 7 > nohup/woman_rain_handbag.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=6 nohup python scripts/test_ctrl_animatediff.py --prompt "a horse galloping on the street, best quality" "a horse galloping on the street with a girl riding on it, best quality" --out_dir "./exp/animatediff/horse_girl/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/horse_girl.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_animatediff.py --prompt "a high quality realistic photo of a cute cat running in a beautiful meadow" "a high quality realistic photo of a cute cat running in a beautiful meadow, Van Gogh style" --out_dir "./exp/animatediff/cat_meadow_van_gogh/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/cat_meadow_van_gogh.txt 2>&1 &`

###### Not So Good

`nohup python scripts/test_ctrl_animatediff.py --prompt "a silver wolf is running" "a silver wolf is running after a golden eagle" --out_dir "./exp/animatediff/wolf_eagle/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 --gpu 1 > nohup/wolf_eagle.txt 2>&1 &`

`nohup python scripts/test_ctrl_animatediff.py --prompt "an astronaut is riding a horse on a road" "an astronaut is riding a horse and holding a Gatling gun on a road" --out_dir "./exp/animatediff/astronaut_horse_gun_road/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 --gpu 2 > nohup/astronaut_horse_gun_road.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_animatediff.py --prompt "Catwoman is holding a sniper rifle" "Catwoman holding a sniper rifle is wearing a hat on her head" --out_dir "./exp/animatediff/catwoman_rifle_hat/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 17 > nohup/catwoman_rifle_hat.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=5 nohup python scripts/test_ctrl_animatediff.py --prompt "a silver wolf is resting on a grassland, with a golden eagle flying high above it" "a silver wolf is resting on a grassland, while a golden eagle is dashing towards it" --out_dir "./exp/animatediff/wolf_change_eagle_position/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/wolf_change_eagle_position.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=6 nohup python scripts/test_ctrl_animatediff.py --prompt "a silver wolf is resting on a grassland" "a silver wolf is resting on a grassland, while a golden eagle is flying towards it" --out_dir "./exp/animatediff/wolf_rest_eagle/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/wolf_rest_eagle.txt 2>&1 &`

##### Remove

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_animatediff.py --prompt "a panda is playing guitar on times square" "a guitar" --out_dir "./exp/animatediff/panda_guitar_rm/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 --ctrl_mode "remove" > nohup/panda_guitar_rm.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=0 nohup python scripts/test_ctrl_animatediff.py --prompt "a panda is playing guitar on times square" "a guitar" --out_dir "./exp/animatediff/panda_guitar_rm_v2/" --src_params 0.9 0.1 2 --tgt_params -1.0 0.1 21 --ctrl_mode "remove" --removal_version 2 > nohup/panda_guitar_rm.txt 2>&1 &`

## 3D

[
    'a horse galloping on the street, best quality',
    'a horse galloping on the street with a girl riding on it, best quality'
]

[
    "a corgi",
    "a corgi wearing a bowler hat",
]

### CLI

#### Shap-E

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_shap_e.py --prompt "a horse galloping on the street, best quality" "a horse galloping on the street with a girl riding on it, best quality" --out_dir "./exp/shap_e/horse_girl/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 16 > nohup/horse_girl.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=7 nohup python scripts/test_ctrl_shap_e.py --prompt "a dog" "a dog wearing a hat" --out_dir "./exp/shap_e/dog_hat/" --src_params 0.9 0.1 2 --tgt_params 0.0 0.1 17 > nohup/dog_hat.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=6 nohup python scripts/test_ctrl_shap_e.py --prompt "a chair" "a chair and a desk" --out_dir "./exp/shap_e/chair_desk/" --src_params 1.0 0.1 2 --tgt_params 0.0 0.1 17 > nohup/chair_desk.txt 2>&1 &`

#### LucidDreamer

`bash scripts/test_ctrl_lucid_dreamer.sh`

##### Add

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=0.7 --workspace_cli='horse_girl_sphere/0.9_0.7' &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/panda_guitar.yaml --w_src_cli=0.9 --w_tgt_cli=0.6 --workspace_cli='panda_guitar/0.9_0.6' &`

`CUDA_VISIBLE_DEVICES=2 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/girl_hat.yaml --w_src_cli=0.9 --w_tgt_cli=0.2 --workspace_cli='girl_hat/0.9_0.2' &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/car_gun.yaml --w_src_cli=1.0 --w_tgt_cli=0.0 --workspace_cli='car_gun/1.0_0.0' &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/motor_flames.yaml --w_src_cli=1.0 --w_tgt_cli=1.3 --workspace_cli='motor_flames/1.0_1.3' &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/burger_fries.yaml --w_src_cli=1.0 --w_tgt_cli=0.2 --workspace_cli='burger_fries/1.0_0.2' &`

##### Remove

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/cat_armor.yaml --w_src_cli=1.0 --w_tgt_cli=0.2 --workspace_cli='cat_armor/1.0_0.2' --ctrl_mode_cli "remove" &`

`CUDA_VISIBLE_DEVICES=2 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/warrior_horse.yaml --w_src_cli=1.0 --w_tgt_cli=0.2 --workspace_cli='warrior_horse/1.0_0.2' --ctrl_mode_cli "remove" &`

`CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/pikachu_crown.yaml --w_src_cli=1.0 --w_tgt_cli=-1.0 --workspace_cli='pikachu_crown/1.0_-1.0' --ctrl_mode_cli "remove" --removal_version_cli=2 &`

#### LGM

`python scripts/test_lgm.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/lgm/samples`

##### Add

`CUDA_VISIBLE_DEVICES=5 python scripts/test_ctrl_lgm.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/lgm/corgi_hat/ --prompts "a corgi" "a corgi wearing a bowler hat" --src_params 0.5 0.2 7 --tgt_params 0.0 0.1 21 > nohup/corgi_hat.txt 2>&1 &`

##### Remove

`CUDA_VISIBLE_DEVICES=7 python scripts/test_ctrl_lgm.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/lgm/cat_armor/ --prompts "a DSLR photo of a cat wearing armor" "an armor" --src_params 0.5 0.1 11 --tgt_params 0.0 0.1 21 --ctrl_mode "remove" > nohup/cat_armor.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=2 python scripts/test_ctrl_lgm.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/lgm/cat_clothes/ --prompts "a white cat wearing blue clothes" "blue clothes" --src_params 0.7 0.1 16 --tgt_params 0.0 0.1 24 --ctrl_mode "remove" > nohup/cat_clothes.txt 2>&1 &`

`CUDA_VISIBLE_DEVICES=7 python scripts/test_ctrl_lgm.py big --resume ctrl_3d/LGM/pretrained/model_fp16_fixrot.safetensors --workspace exp/lgm/cat_clothes/ --prompts "a white cat wearing blue clothes" "blue clothes" --src_params 0.6 0.1 14 --tgt_params -1.0 0.1 24 --ctrl_mode "remove" --removal_version 2 > nohup/cat_clothes.txt 2>&1 &`
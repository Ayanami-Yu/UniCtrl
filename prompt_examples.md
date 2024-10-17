# Prompts Examples

## Image

[
    "Catwoman holding a sniper rifle",
    "Catwoman holding a sniper rifle and wearing a hat",
]
v1:
s = 1.3-1.4
t = 1.0-3.0

v2:
s = 1.7
t = 0.6, 1.3

[
    "an astronaut riding a horse",
    "an astronaut riding a horse and holding a Gatling gun",
]
v1:
s = 0.9
t = 0.5-1.5

## Video

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

### CLI

`nohup python test_video.py --prompt "a silver wolf is running" "a silver wolf is running after a golden eagle" --out_dir "./exp/video/wolf_eagle/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 0 > nohup/wolf_eagle.txt 2>&1 &`

`nohup python test_video.py --prompt "an astronaut is riding a horse on a road" "an astronaut is riding a horse and holding a Gatling gun on a road" --out_dir "./exp/video/astronaut_horse_gun_road/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 1 > nohup/astronaut_horse_gun_road.txt 2>&1 &`

`nohup python test_video.py --prompt "an astronaut is riding a horse" "an astronaut holding a Gatling gun is riding a horse" --out_dir "./exp/video/astronaut_horse_gun/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 2 > nohup/astronaut_horse_gun.txt 2>&1 &`

`nohup python test_video.py --prompt "Catwoman is holding a sniper rifle" "Catwoman holding a sniper rifle is wearing a hat on her head" --out_dir "./exp/video/catwoman_rifle_hat/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 3 > nohup/catwoman_rifle_hat.txt 2>&1 &`

`nohup python test_video.py --prompt "a panda is playing guitar on times square" "a panda is playing guitar on times square, with a drum next to it" --out_dir "./exp/video/panda_guitar_drum/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 4 > nohup/panda_guitar_drum.txt 2>&1 &`

`nohup python test_video.py --prompt "a high quality realistic photo of a cute cat running in a beautiful meadow" "a high quality realistic photo of a cute cat with large wings running in a beautiful meadow" --out_dir "./exp/video/cat_meadow_wings/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 5 > nohup/cat_meadow_wings.txt 2>&1 &`

`nohup python test_video.py --prompt "a woman is walking in the rain" "a woman is walking in the rain and carrying a red handbag" --out_dir "./exp/video/woman_rain_handbag/" --src_params 0.5 0.1 31 --tgt_params 0.1 0.1 41 --gpu 6 > nohup/woman_rain_handbag.txt 2>&1 &`
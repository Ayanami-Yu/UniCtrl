import yaml


config_file = 'metrics/images_v2.yaml'

prompts = [
    "the Eiffel Tower in background",
    "some kids playing around the campfire",
    "a helicopter following the car",
    "sunglasses",
    "sunset, animated style",
    "sunset, oil painting",
    "sunset, pixel art",
    "sunglasses",
    "a soccer ball",
    "rain, a lot of fog behind the Iron Man",
    "a handlebar mustache",
    "a pink scarf",
    "in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy",
    "a delightful and little Tachikoma robot",
    "bright lights, busy traffic",
    "blue and red stripes",
    "A richly textured oil painting.",
    "White flowers.",
    "white hair, deep wrinkles",
    "low-poly game art style",
    "a large open book",
]

names = [
    "boat_tower",
    "campfire_kids",
    "car_helicopter",
    "cat_sunglasses",
    "corgi_animated",
    "corgi_painting",
    "corgi_pixel",
    "corgi_sunglasses",
    "dog_soccer",
    "iron_man_fog",
    "man_mustache",
    "man_scarf",
    "robot_rain",
    "sakura_robot",
    "street_traffic",
    "train_stripes",
    "badger_painting",
    "cow_flowers",
    "lady_hair_wrinkles",
    "rabbit_low_poly",
    "witch_book",
]

with open(config_file, 'r') as f:
    data = yaml.safe_load(f)

for mode in data.keys():
    for name in data[mode].keys():
        for i in range(len(names)):
            if names[i] == name:
                data[mode][name]['tgt_prompt']['change'] = prompts[i]

with open(config_file, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
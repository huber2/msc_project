from os.path import dirname, abspath
import json


test_experinece_params = {
    "scene_file": "Franka_2cuboids_32camera.ttt",
    "object_names": ["cuboid0", "cuboid1"],
    "task_args":
    {
        "n_tests": 100,
        "max_steps": 300,
        "distance_tolerance": 0.05,
        "maintain_target_duration": 10,
    },
    "random_seed": 123,
    "models_suffix": ["_aug_model"],
    "colors_tested": 
    {
        "FF0000": 
        {
            'target_color': [255, 0, 0],
            'distractor_colors': [
                [0, 0, 0], 
                [255, 255, 255], 
                [255, 255, 0], 
                [255, 0, 255], 
                [128, 0, 0], 
                [192, 0, 0], 
                [230, 0, 0], 
                [255, 0, 0], 
                [255, 128, 128], 
                [256, 26, 26]
                ]
        },
        "00FF00": 
        {
            'target_color': [0, 255, 0],
            'distractor_colors': [
                [0, 0, 0], 
                [255, 255, 255], 
                [255, 255, 0], 
                [0, 255, 255], 
                [0, 128, 0], 
                [0, 192, 0], 
                [0, 230, 0], 
                [0, 255, 0], 
                [128, 255, 128], 
                [26, 256, 26]
                ]
        },
        "0000FF": 
        {
            'target_color': [255, 0, 0],
            'distractor_colors': [
                [0, 0, 0], 
                [255, 255, 255], 
                [255, 0, 255], 
                [0, 255, 255], 
                [0, 0, 128], 
                [0, 0, 192], 
                [0, 0, 230], 
                [0, 0, 255], 
                [128, 128, 255], 
                [26, 26, 256]
                ]
        },
    },
}


if __name__=='__main__':
    DIR_PATH = dirname(abspath(__file__)) + '/../'
    with open(f"{DIR_PATH}data/test_parameters.json", 'w') as f:
        json.dump(test_experinece_params, f, indent=4)
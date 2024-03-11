import os
from constants import *
from utils import get_original_splits, get_scene_name_to_location
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the nuscenes dataset directory")
    parser.add_argument("--f_no_original_test_scenes", type=bool, help="Flag to remove the original test scenes from the new splits", default=False)
    args = parser.parse_args()

    data_dir = args.data_dir
    f_no_original_test_scenes = args.f_no_original_test_scenes
    
    scene_name_to_location = get_scene_name_to_location(data_dir)

    original_test_splits = get_original_splits()["test"]

    for split_name, map_names in CITYWISE_SPLITS.items():

        if f_no_original_test_scenes:
            split_name += "_no_original_test_scenes"

        print(f"Creating new splits for {split_name}")
        out_folder = f"{SPLIT_PICKLES_ROOT}/city_splits"
        os.makedirs(out_folder, exist_ok=True)

        scenes_in_split = []
        for scene, location in scene_name_to_location.items():
            if f_no_original_test_scenes and scene in original_test_splits:
                continue

            if location in map_names:
                scenes_in_split.append(scene)

        # save list of scenes in split as txt file
        with open(f"{out_folder}/{split_name}.txt", "w") as f:
            for scene in scenes_in_split:
                f.write(f"{scene}\n")




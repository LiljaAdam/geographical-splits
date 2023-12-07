from nuscenes import NuScenes
import os
import pickle

def get_original_splits():
    from nuscenes.utils import splits

    original_splits = splits.create_splits_scenes()
    original_splits.pop("train_detect")
    original_splits.pop("train_track")
    original_splits.pop("mini_train")
    original_splits.pop("mini_val")

    scene_to_original_split = {}
    for split_name, scene_names in original_splits.items():
        for scene_name in scene_names:
            scene_to_original_split[scene_name] = split_name

    return scene_to_original_split

def get_token_to_scene_mapping(pkl_root):
    token_to_scene_name_file = os.path.join("new_splits_nusc", "token_to_scene_name.pkl")
    if os.path.exists(token_to_scene_name_file):
        print("Loading token_to_scene_name from: ", token_to_scene_name_file)
        with open(token_to_scene_name_file, "rb") as f:
            token_to_scene_name = pickle.load(f)
        return token_to_scene_name

    else:
        print("Loading NuScenes...")
        nusc = NuScenes(
            version="v1.0-trainval",
            dataroot=pkl_root,
        )

        nusc_test = NuScenes(
            version="v1.0-test",
            dataroot=pkl_root,
        )

        token_to_scene_name = {}
        for scene in nusc.scene:
            token_to_scene_name[scene["token"]] = scene["name"]

        for scene in nusc_test.scene:
            token_to_scene_name[scene["token"]] = scene["name"]

        print("Saving token_to_scene_name to: ", token_to_scene_name_file)
        with open(token_to_scene_name_file, "wb") as f:
            pickle.dump(token_to_scene_name, f)

        return token_to_scene_name
import pickle
import os
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
from constants import *
import numpy as np
from scipy.spatial.transform import Rotation
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

def fetch_from_nusc(scenes_positions, nusc):
    for sample in nusc.sample:
        sample_token = sample["token"]
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        location = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        lidar_token = sample["data"]["LIDAR_TOP"]
        ego_pose = nusc.get("ego_pose", nusc.get("sample_data", lidar_token)["ego_pose_token"])
        positions = get_positions(ego_pose, flat=True)

        if scene_name not in scenes_positions[location]:
            scenes_positions[location][scene_name] = {"positions": [], 
                                                      "tokens": []}

        scenes_positions[location][scene_name]["positions"].append(Point(positions))
        scenes_positions[location][scene_name]["tokens"].append(sample_token)
    
    return scenes_positions

def create_sample_positions(data_dir, data_splits, map_names):
    scenes_positions = {map_name: {} for map_name in map_names}

    nusc = NuScenes(
                version="v1.0-trainval",
                dataroot=data_dir,
            )
    nusc_test = NuScenes(
                version="v1.0-test",
                dataroot=data_dir,
            )

    scenes_positions = fetch_from_nusc(scenes_positions, nusc)
    scenes_positions = fetch_from_nusc(scenes_positions, nusc_test)

    return scenes_positions


def get_samples_positions(data_dir=None, data_splits=None, map_names=None):
    if data_splits is None:
        data_splits = ["train", "val", "test"]

    if map_names is None:
        map_names = CITY_MAPS

    if not os.path.exists(SAMPLES_POSITIONS_PICKLE):
        print("Creating new samples positions pickle")
        samples_positions = create_sample_positions(data_dir, data_splits, map_names)
        print(f"Saving sample positions to {SAMPLES_POSITIONS_PICKLE}")
        os.makedirs(os.path.dirname(SAMPLES_POSITIONS_PICKLE), exist_ok=True)
        with open(SAMPLES_POSITIONS_PICKLE, "wb") as f:
            pickle.dump(samples_positions, f)
    else:
        print("Loading samples positions from pickle")
        with open(SAMPLES_POSITIONS_PICKLE, "rb") as f:
            samples_positions = pickle.load(f)

    return samples_positions


def get_original_splits():
    from nuscenes.utils import splits

    original_splits = splits.create_splits_scenes()
    original_splits.pop("train_detect")
    original_splits.pop("train_track")
    original_splits.pop("mini_train")
    original_splits.pop("mini_val")

    # remove scenes that are in LOG_IDS_BLACKLIST from splits
    for split_name, scene_names in original_splits.items():
        scene_names = [
            scene_name
            for scene_name in scene_names
            if scene_name not in LOG_IDS_BLACKLIST
        ]
        original_splits[split_name] = scene_names

    return original_splits


def get_new_splits(new_split_path="new_splits_nusc/new_splits.pickle", split_name=None):
    if split_name is not None:
        new_split_path = (
            f"new_splits_nusc/versions/{split_name}/new_splits_{split_name}.pickle"
        )

    print("Loading new splits from: ", new_split_path)
    with open(new_split_path, "rb") as f:
        new_splits = pickle.load(f)
    return new_splits


def get_scene_to_location_mapping(pkl_path="new_splits_nusc/scene_to_location.pkl"):
    assert os.path.exists(
        pkl_path
    ), f"Path {pkl_path} does not exist. Run 'create_new_splits.py' first"

    with open(pkl_path, "rb") as f:
        scene_to_location = pickle.load(f)
    return scene_to_location


def create_scene_name_to_split_mapping(splits):
    scene_name_to_split = {}
    for split_name, scene_names in splits.items():
        for scene_name in scene_names:
            scene_name_to_split[scene_name] = split_name
    return scene_name_to_split


def get_nuscene_maps(map_names, dataroot):
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc_maps = {
        map_name: NuScenesMap(
            dataroot=dataroot,
            map_name=map_name,
        )
        for map_name in map_names
    }

    return nusc_maps


def get_night_rain_info_mapping():
    assert os.path.exists(
        NIGHT_RAIN_INFO_PICKLE
    ), f"Path {NIGHT_RAIN_INFO_PICKLE} does not exist. Run 'create_night_rain_info.py' first"

    with open(NIGHT_RAIN_INFO_PICKLE, "rb") as f:
        night_rain_info = pickle.load(f)
    return night_rain_info


def get_sample_token_to_scene_name_mapping():
    with open(SAMPLE_TOKEN_TO_SCENE_NAME_PICKLE, "rb") as f:
        sample_token_to_scene_name = pickle.load(f)
    return sample_token_to_scene_name

def get_sample_token_to_location():
    with open(SAMPLE_TOKEN_TO_LOCATION_PICKLE, "rb") as f:
        sample_token_to_location = pickle.load(f)
    return sample_token_to_location


def get_scene_name_to_sample_tokens():
    sample_token_to_scene_name = get_sample_token_to_scene_name_mapping()
    scene_name_to_sample_tokens = {}
    for sample_token, scene_name in sample_token_to_scene_name.items():
        if scene_name not in scene_name_to_sample_tokens:
            scene_name_to_sample_tokens[scene_name] = []
        scene_name_to_sample_tokens[scene_name].append(sample_token)
    return scene_name_to_sample_tokens


def get_sample_to_split_from_scene_to_split(splits, scene_name_to_sample_tokens):
    sample_to_split = {}
    for split, scene_names in splits.items():
        for scene_name in scene_names:
            sample_tokens = scene_name_to_sample_tokens[scene_name]
            for sample_token in sample_tokens:
                sample_to_split[sample_token] = split
    return sample_to_split


def get_split_to_samples(sample_to_split):
    split_to_samples = {}
    for sample, split in sample_to_split.items():
        if split not in split_to_samples:
            split_to_samples[split] = []
        split_to_samples[split].append(sample)
    return split_to_samples


def get_scene_name_to_night_rain_info():
    with open(NIGHT_RAIN_INFO_PICKLE, "rb") as f:
        night_rain_info = pickle.load(f)
    return night_rain_info


def get_sample_to_num_objects(range=None):
    file = SAMPLES_TO_NUM_OBJECTS_PICKLE
    if range is not None:
        file = file.replace(".pkl", f"_{range}m.pkl")
    with open(file, "rb") as f:
        sample_to_num_objects = pickle.load(f)
    return sample_to_num_objects


def get_closest_mapping(split_name, threshold_range_m=None, threshold_heading_deg=None):
    val_name = "closest_mapping_per_map_val"
    test_name = "closest_mapping_per_map_test"

    if threshold_heading_deg is not None:
        if type(threshold_heading_deg) == float:
            threshold_heading_deg = str(threshold_heading_deg).replace(".", "_")

        val_name += f"_h{threshold_heading_deg}"
        test_name += f"_h{threshold_heading_deg}"

    if threshold_range_m is not None:
        if type(threshold_range_m) == float:
            threshold_range_m = str(threshold_range_m).replace(".", "_")

        val_name += f"_r{threshold_range_m}"
        test_name += f"_r{threshold_range_m}"


    file_path_val = (
        CLOSEST_MAPPINGS + split_name + f"/{val_name}.pickle"
    )
    file_path_test = (
        CLOSEST_MAPPINGS + split_name + f"/{test_name}.pickle"
    )

    with open(file_path_val, "rb") as f:
        closest_mapping_val = pickle.load(f)
    with open(file_path_test, "rb") as f:
        closest_mapping_test = pickle.load(f)

    # combine the two dictionaries without location
    closest_mapping = {}
    for map_name, closest_mapping_per_map in closest_mapping_val.items():
        for val_sample, train_sample in closest_mapping_per_map.items():
            assert val_sample not in closest_mapping
            closest_mapping[val_sample] = train_sample

    for map_name, closest_mapping_per_map in closest_mapping_test.items():
        for test_sample, train_sample in closest_mapping_per_map.items():
            assert test_sample not in closest_mapping
            closest_mapping[test_sample] = train_sample

    return closest_mapping


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t
    return pose


def get_positions(ego_pose, inv=False, flat=False):
    translation = ego_pose["translation"]
    rotation = ego_pose["rotation"]
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix
    t = np.array(translation, dtype=np.float32)
    return get_transformation_matrix(R, t, inv=inv)[:2, 3]
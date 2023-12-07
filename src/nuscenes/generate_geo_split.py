import plotly.graph_objects as go
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
import pickle
import os
from nuscenes.utils import splits
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from constants import *
from utils import get_samples_positions, get_original_splits
import yaml
import argparse

def get_nuscene_maps(map_names, data_dir):
    nusc_maps = {
        map_name: NuScenesMap(
            dataroot=data_dir,
            map_name=map_name,
        )
        for map_name in map_names
    }
    return nusc_maps

def get_map_polygons_from_file(split_coordinates_file):
    # get the polygons from the yaml file and convert them to shapely polygons
    # the yaml file is on the format:
    #  map_name:
    #    val:
    #       - polygon1 (list)
    #       - polygon2 (list)
    #       - ...
    #    test:
    #       - polygon1 (list)
    #       - polygon2 (list)
    #       - ...

    all_map_polygons = {}
    with open(split_coordinates_file.replace("_no_original_test_scenes", ""), "r") as f:
        split_coordinates = yaml.safe_load(f)

    for map_name, splits in split_coordinates.items():
        all_map_polygons[map_name] = {}
        for split, polygons in splits.items():
            all_map_polygons[map_name][split] = [
                Polygon(np.array(poly)) for poly in polygons if len(poly) > 0
            ]

    return all_map_polygons

def save_map_visualization(
    map_names,
    nusc_maps,
    token_to_new_split,
    sample_infos,
    all_map_polygons,
):
    for map_name in map_names:
        nusc_map = nusc_maps[map_name]

        if F_RENDER_MAP:
            fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        for scene_name, infos in sample_infos[map_name].items():
            
            positions = np.array([pos.xy for pos in infos["positions"]])
            positions = positions.squeeze(axis=-1)

            splits = [token_to_new_split[token] for token in infos["tokens"]]

            colors = [COLORS[split] for split in splits]

            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                color=colors,
                s=1,
                zorder=2,
                alpha=0.5,
            )


        for polygon in all_map_polygons[map_name]['val']:
            ax.plot(
                *polygon.exterior.xy,
                color=COLORS["val"],
                linewidth=4,
                zorder=2,
                alpha=0.5,
                linestyle="dotted",
            )

        for polygon in all_map_polygons[map_name]['test']:
            ax.plot(
                *polygon.exterior.xy,
                color=COLORS["test"],
                linewidth=4,
                zorder=2,
                alpha=0.5,
                linestyle="dotted",
            )

        # remove background to be transparant
        ax.set_facecolor("none")
        
        fig_path = f"{FIGURES_PATH}/{split_name}"
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(
            f"{fig_path}/map_{map_name}.png", bbox_inches="tight", dpi=300, pad_inches=0, transparent=True
        )


def create_new_fine_splits(
    samples, map_names, all_map_polygons
):
    split_assignment_per_sample_token = {}
    for map_name in map_names:
        map_polygons_val = all_map_polygons[map_name]['val']
        map_polygons_test = all_map_polygons[map_name]['test']

        split_assignment = {
            scene_name: None for scene_name in samples[map_name].keys()
        }

        for scene_name, sample in samples[map_name].items():
            token_assignment = {
                token: None for token in sample["tokens"]
            }
            for position, token in zip(sample["positions"], sample["tokens"]):
                has_not_been_assigned = True
                for polygon in map_polygons_val:
                    if polygon.contains(position) and has_not_been_assigned:
                        token_assignment[token] = "val"
                        has_not_been_assigned = False

                for polygon in map_polygons_test:
                    if polygon.contains(position) and has_not_been_assigned:
                        token_assignment[token] = "test"
                        has_not_been_assigned = False

                if has_not_been_assigned:
                    token_assignment[token] = "train"
                    has_not_been_assigned = False

            split_assignment[scene_name] = token_assignment


        split_assignment_per_sample_token[map_name] = split_assignment

    return split_assignment_per_sample_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the nuscenes dataset directory")
    parser.add_argument("--f_no_original_test_scenes", type=bool, help="Flag to remove the original test scenes from the new splits", default=False)
    parser.add_argument("--f_render_map", type=bool, help="Flag to render the map", default=False)
    parser.add_argument("--f_save_map_visualization", type=bool, help="Flag to save the map visualization", default=True)
    parser.add_argument("--split_name", type=str, help="Name of the split", default="nusc_geo")
    args = parser.parse_args()

    split_name = args.split_name
    data_dir = args.data_dir
    f_no_original_test_scenes = args.f_no_original_test_scenes
    F_RENDER_MAP = args.f_render_map
    f_save_map_visualization = args.f_save_map_visualization

    if f_no_original_test_scenes:
        split_name += "_no_original_test_scenes"

    print(f"Creating new splits for {split_name}")
    split_coordinates_file = os.path.join(ZONE_COORDINATES, f"{split_name}.yaml")
    OUTPUT_ROOT = f"{SPLIT_PICKLES_ROOT}/{split_name}"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    data_splits = ["train", "val", "test"]

    map_names = [
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
    ]

    sample_infos = get_samples_positions(data_dir, data_splits, map_names)

    if f_no_original_test_scenes:
        original_test_splits = get_original_splits()["test"]
        filtered_sample_infos = {}
        for map_name in map_names:
            filtered_sample_infos[map_name] = {}
            for scene_name in sample_infos[map_name].keys():
                if scene_name not in original_test_splits:
                    filtered_sample_infos[map_name][scene_name] = sample_infos[map_name][scene_name]
        sample_infos = filtered_sample_infos

    nusc_maps = get_nuscene_maps(map_names, data_dir)
    all_map_polygons = get_map_polygons_from_file(split_coordinates_file)

    print("Creating new splits")
    split_assignment_per_sample_token = create_new_fine_splits(
        sample_infos, map_names, all_map_polygons
    )

    # create mapping from token to new split
    token_to_new_split = {}
    for map_name in map_names:
        for scene_name, scene_info in split_assignment_per_sample_token[map_name].items():
            for token, split in scene_info.items():
                token_to_new_split[token] = split

    print(f"Saving {split_name} splits")
    with open(f"{OUTPUT_ROOT}/{split_name}.pkl", "wb") as f:
        pickle.dump(token_to_new_split, f)

    if f_save_map_visualization:
        print("Saving map visualizations")
        save_map_visualization(
            map_names,
            nusc_maps,
            token_to_new_split,
            sample_infos,
            all_map_polygons,
        )




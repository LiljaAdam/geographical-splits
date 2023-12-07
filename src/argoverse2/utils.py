from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import plotly.graph_objects as go
import pickle
import os
import os.path as osp
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
import yaml
import numpy as np
OUR_SPLIT_NAME = "Geo."
ORIGINAL_SPLIT_NAME= "Orig."
PON_SPLIT_NAME = "Coarse"
FAIL_LOGS = [
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
]
COLORS = {"train": "g", "val": "b", "test": "r"}
ALPHAS = {"train": 1.0, "val": 1.0, "test": 1.0}
HTML_COLORS = {"train": "#33CC00", "val": "#6666FF", "test": "#CC3333", "removed": "black"}
CAM_NAMES = ['ring_front_center',  'ring_front_left', 'ring_front_right',  'ring_side_left',  'ring_side_right', 'ring_rear_right', 'ring_rear_left']

MAP_NAMES = ['PIT', 'PAO', 'ATX', 'WDC', 'MIA', 'DTW']

DATASET_NAME = "argoverse2"
DATASET_DIR = f"src/{DATASET_NAME}"
ZONE_COORDINATES = f"{DATASET_DIR}/zone_coordinates/"

HELPER_DIR = f"{DATASET_DIR}/helper_pkls/"

FIGURE_ROOT = f"{DATASET_DIR}/figures"
HEATMAPS_ROOT = f"{HELPER_DIR}/heatmaps"
SPLIT_ROOT = f"{DATASET_DIR}/splits"
CLOSE_MAPPINGS = f"{HELPER_DIR}/close_mappings"
CLOSEST_MAPPINGS = f"{HELPER_DIR}/closest_mappings"

def get_valid_log_ids(av2 : AV2SensorDataLoader):
    log_ids = list(av2.get_log_ids())
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)
    return log_ids

def get_log_data(data_root, f_force_reload=False):

    log_id_to_split_path = Path(HELPER_DIR + "/log_id_to_split.pkl")
    log_id_to_location_path = Path(HELPER_DIR + "/log_id_to_location.pkl")
    log_id_to_poses_path = Path(HELPER_DIR + "/log_id_to_poses.pkl")
    log_id_to_headings_path = Path(HELPER_DIR + "/log_id_to_headings.pkl")
    log_id_to_sample_ids_path = Path(HELPER_DIR + "/log_id_to_sample_ids.pkl")

    if not f_force_reload and log_id_to_split_path.exists() and log_id_to_location_path.exists() and log_id_to_poses_path.exists() and log_id_to_headings_path.exists():
        print("Loading log data from pickles")
        with open(log_id_to_split_path, "rb") as f:
            log_id_to_split = pickle.load(f)
        with open(log_id_to_location_path, "rb") as f:
            log_id_to_location = pickle.load(f)
        with open(log_id_to_poses_path, "rb") as f:
            log_id_to_poses = pickle.load(f)
        with open(log_id_to_headings_path, "rb") as f:
            log_id_to_headings = pickle.load(f)
        with open(log_id_to_sample_ids_path, "rb") as f:
            log_id_to_sample_ids = pickle.load(f)
    
    else:
        print("Loading log data from scratch")
        train_path = Path(data_root + "/train")
        val_path = Path(data_root + "/val")
        test_path = Path(data_root + "/test")
        av2_train = AV2SensorDataLoader(data_dir=train_path, labels_dir=train_path)
        av2_val = AV2SensorDataLoader(data_dir=val_path, labels_dir=val_path)
        av2_test = AV2SensorDataLoader(data_dir=test_path, labels_dir=test_path)
        av2s = {"train": av2_train, "val": av2_val, "test": av2_test}

        # Load the information from each original split
        log_id_to_split = {}
        log_id_to_location = {}
        log_id_to_poses = {}
        log_id_to_headings = {}
        log_id_to_sample_ids = {}
        for split, av2 in av2s.items():
            log_ids = get_valid_log_ids(av2)

            for log_id in log_ids:
                
                log_id_to_split[log_id] = split
                log_id_to_location[log_id] = av2.get_city_name(log_id)

                poses, headings, sample_ids = get_poses(av2, log_id)
                log_id_to_poses[log_id] = poses 
                log_id_to_headings[log_id] = headings
                log_id_to_sample_ids[log_id] = sample_ids

        # Create folder if not exists
        if not os.path.exists(HELPER_DIR):
            os.makedirs(HELPER_DIR)

        # Save the information to pickles
        with open(log_id_to_split_path, "wb") as f:
            pickle.dump(log_id_to_split, f)
        with open(log_id_to_location_path, "wb") as f:
            pickle.dump(log_id_to_location, f)
        with open(log_id_to_poses_path, "wb") as f:
            pickle.dump(log_id_to_poses, f)
        with open(log_id_to_headings_path, "wb") as f:
            pickle.dump(log_id_to_headings, f)
        with open(log_id_to_sample_ids_path, "wb") as f:
            pickle.dump(log_id_to_sample_ids, f)

    return log_id_to_split, log_id_to_location, log_id_to_poses, log_id_to_headings, log_id_to_sample_ids

def get_poses(av2: AV2SensorDataLoader, log_id: str):
    discarded = 0
    lidar_timestamps = av2._sdb.per_log_lidar_timestamps_index[log_id]
    poses = []
    headings = []
    sample_ids = [] 
    for ts in lidar_timestamps:
        cam_ring_fpath = [av2.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = av2.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        city_SE3_ego = av2.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation

        # Get the heading
        heading_rad = np.arctan2(e2g_rotation[1, 0], e2g_rotation[0, 0])
        heading = np.rad2deg(heading_rad)
        headings.append(heading)

        poses.append(e2g_translation)

        pts_filename = osp.basename(lidar_fpath)
        pts_filename = pts_filename.split('.')[0]
        sample_ids.append(pts_filename)

    return poses, headings, sample_ids

def visualize_log_data(log_id_to_split, log_id_to_location, log_id_to_poses, split_name="original", map_polygons=None):
    # get unique locations
    locations = list(set(log_id_to_location.values()))

    # create one figure for each location and put in a dictionary
    location_to_fig = {}
    for location in locations:
        location_to_fig[location] = go.Figure()

        if map_polygons is not None:
            for split, polygons in map_polygons[location].items():
                for polygon in polygons:
                    location_to_fig[location].add_trace(go.Scatter(
                        x = list(polygon.exterior.xy[0]),
                        y = list(polygon.exterior.xy[1]),
                        mode = 'lines',
                        name = split,
                        line=dict(color=HTML_COLORS[split], width=5, dash='dot'),
                        opacity = 0.5,
                    )
                    )


    for log_id, poses in log_id_to_poses.items():
        location = log_id_to_location[log_id]
        split = log_id_to_split[log_id]

        xpos = [pose[0] for pose in poses]
        ypos = [pose[1] for pose in poses]
        # add the poses to the figure
        location_to_fig[location].add_trace(go.Scatter(
            x = xpos,
            y = ypos,
            mode = 'lines',
            name = split,
            line=dict(color=HTML_COLORS[split], width=7),
            opacity = ALPHAS[split],
        )
        )
    
    # create the png figure folder
    out_folder = f"{FIGURE_ROOT}/{split_name}/split_poses"
    os.makedirs(out_folder, exist_ok=True)

    # save the figures
    for location, fig in location_to_fig.items():
        fig.write_image(f"{out_folder}/{location}.png", width=1920, height=1080)


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
    with open(split_coordinates_file, "r") as f:
        split_coordinates = yaml.safe_load(f)

    for map_name, splits in split_coordinates.items():
        all_map_polygons[map_name] = {}
        for split, polygons in splits.items():
            all_map_polygons[map_name][split] = [
                Polygon(np.array(poly)) for poly in polygons if len(poly) > 0
            ]

    return all_map_polygons


def save_new_splits(log_id_to_new_split, split_name):
    # Create folder if not exists
    out_folder = f"{SPLIT_ROOT}/{split_name}"
    os.makedirs(out_folder, exist_ok=True)

    # Save the information to pickles
    with open(f"{out_folder}/{split_name}.pkl", "wb") as f:
        pickle.dump(log_id_to_new_split, f)

def load_new_splits(split_name):

    if split_name == "original":
        log_id_to_split, _, _, _ = get_log_data()
        return log_id_to_split

    in_folder = f"{SPLIT_ROOT}/{split_name}"
    pkl_file = f"{split_name}.pkl"

    pkl_path = os.path.join(in_folder, pkl_file)

    if not os.path.exists(pkl_path):
        pkl_file = f"{split_name}.pkl"
    
    pkl_path = os.path.join(in_folder, pkl_file)

    with open(pkl_path, "rb") as f:
        log_id_to_split = pickle.load(f)

    return log_id_to_split

def get_cvprc_to_av2_mapping():
    with open(f"{PICKLE_ROOT}/cvprc_to_av2.pkl", "rb") as f:
        cvprc_to_av2 = pickle.load(f)
    return cvprc_to_av2

def get_av2_to_cvprc_mapping():
    cvprc_to_av2 = get_cvprc_to_av2_mapping()
    av2_to_cvprc = {v: k for k, v in cvprc_to_av2.items()}
    return av2_to_cvprc
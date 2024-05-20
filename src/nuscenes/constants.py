OUR_SPLIT_NAME = "Geo."
ORIGINAL_SPLIT_NAME= "Orig."
PON_SPLIT_NAME = "Coarse"

COLORS = {"train": "g", "val": "b", "test": "r"}
ALPHAS = {"train": 0.8, "val": 0.3, "test": 0.3, "removed": 0.1}

HTML_COLORS = {"train": "#33CC00", "val": "#6666FF", "test": "#CC3333", "removed": "black"}
INDEX_COLORS = [ "green","blue","red","orange","purple","brown","pink","gray","olive","cyan"]

DATASET_NAME = "nuscenes"
DATASET_DIR = f"src/{DATASET_NAME}"
ZONE_COORDINATES = f"{DATASET_DIR}/zone_coordinates/"

HELPER_DIR = f"{DATASET_DIR}/helper_pkls/"
SCENES_POSITIONS_PICKLE = f"{HELPER_DIR}/scenes_positions.pkl"
SCENES_HEADINGS_PICKLE = f"{HELPER_DIR}/scenes_headings.pkl"
SAMPLES_POSITIONS_PICKLE = f"{HELPER_DIR}/samples_positions.pkl"
SAMPLES_HEADINGS_PICKLE = f"{HELPER_DIR}/samples_headings.pkl"
SCENE_TOKEN_TO_SCENE_NAME_PICKLE = f"{HELPER_DIR}/scene_token_to_scene_name.pkl"
SAMPLE_TOKEN_TO_SCENE_NAME_PICKLE = f"{HELPER_DIR}/sample_token_to_scene_name.pkl"
SAMPLE_TOKEN_TO_LOCATION_PICKLE = f"{HELPER_DIR}/sample_token_to_location.pkl"
NIGHT_RAIN_INFO_PICKLE = f"{HELPER_DIR}/scene_to_night_rain_info.pkl"
SAMPLES_TO_NUM_OBJECTS_PICKLE = f"{HELPER_DIR}/sample_token_to_num_objects.pkl"
SCENE_TO_LOCATION_PICKLE = f"{HELPER_DIR}/scene_to_location.pkl"

CLOSE_MAPPINGS = f"{DATASET_DIR}/close_mappings/"
CLOSEST_MAPPINGS = f"{DATASET_DIR}/closest_mappings/"
FIGURES_PATH = f"{DATASET_DIR}/figures/"
SPLIT_PICKLES_ROOT = f"{DATASET_DIR}/splits/"
HEATMAPS_ROOT = f"{DATASET_DIR}/num_samples_heatmaps/"

CLASS_MAP_RGE_TO_NUSC = {
    "lane_markers": ["lane_divider", "road_divider"],
    "road_edges": ["road_segment", "lane"],
    "ped_crossings": ["ped_crossing"],
}

CITY_MAPS = [
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
]

MAP_NAMES = CITY_MAPS

LOG_IDS_BLACKLIST = [
    "scene-0499",
    "scene-0064",
    "scene-0502",
    "scene-0515",
    "scene-0517",
]

CITYWISE_SPLITS = {
                    "hollandsvillage": ["singapore-hollandvillage"],
                    "queenstown": ["singapore-queenstown"],
                    "boston": ["boston-seaport"],
                    "singapore": ["singapore-onenorth", 
                                  "singapore-hollandvillage",
                                  "singapore-queenstown"],
                    "boston+onenorth": ["boston-seaport", 
                                        "singapore-onenorth"],
                    "hollandsvillage+queenstown": ["singapore-hollandvillage", 
                                                   "singapore-queenstown"],
                    "boston+hollandsvillage+queenstown": ["boston-seaport", 
                                                          "singapore-hollandvillage", 
                                                          "singapore-queenstown"],
                    "onenorth": ["singapore-onenorth"],
                }
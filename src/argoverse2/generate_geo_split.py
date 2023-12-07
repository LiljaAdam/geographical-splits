
from utils import *
import argparse

def create_new_splits(log_id_to_location, log_id_to_poses, all_map_polygons):
    log_id_to_new_split = {}

    for log_id, poses in log_id_to_poses.items():
        location = log_id_to_location[log_id]

        # Get the map polygons for this location
        location_map_polygons = all_map_polygons[location]

        # Check if all poses are within a map polygon
        ls = LineString(poses)
        f_in_polygon = False

        for split, polygons in location_map_polygons.items():
            for polygon in polygons:
                if polygon.contains(ls):
                    log_id_to_new_split[log_id] = split
                    f_in_polygon = True
                    break
            if not f_in_polygon:
                log_id_to_new_split[log_id] = "train"
                
    return log_id_to_new_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the argoverse dataset directory")
    parser.add_argument("--f_save_map_visualization", type=bool, help="Flag to save the map visualization", default=True)
    parser.add_argument("--split_name", type=str, help="Name of the split", default="argo_geo")

    args = parser.parse_args()

    split_name = args.split_name
    data_dir = args.data_dir
    f_save_map_visualization = args.f_save_map_visualization
    split_coordinates_file = ZONE_COORDINATES + split_name + ".yaml"

    # Load the original data
    log_id_to_split, log_id_to_location, log_id_to_poses, _, _ = get_log_data(data_root=data_dir)

    if f_save_map_visualization:
        visualize_log_data(log_id_to_split, log_id_to_location, log_id_to_poses)

    # Get the map polygons
    map_polygons = get_map_polygons_from_file(split_coordinates_file)

    # Get the new splits according to defined polygons
    print("Creating new splits")
    log_id_to_new_split = create_new_splits(
        log_id_to_location, log_id_to_poses, map_polygons
    )

    # Save the new splits
    print("Saving new splits")
    save_new_splits(log_id_to_new_split, split_name)

    if f_save_map_visualization:
        print("Saving map visualization")
        visualize_log_data(log_id_to_new_split, log_id_to_location, log_id_to_poses, split_name, map_polygons)
        


    


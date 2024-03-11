
from utils import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the argoverse dataset directory")

    args = parser.parse_args()
    data_dir = args.data_dir

    out_folder = f"{SPLIT_ROOT}/city_splits"
    os.makedirs(out_folder, exist_ok=True)
    
    # Load the original data
    _, log_id_to_location, _, _, _ = get_log_data(data_root=data_dir)

    for split_name, cities in CITYWISE_SPLITS.items():
        print(f"Creating new splits for {split_name}")

        log_ids_in_split = [log_id for log_id, location in log_id_to_location.items() if location in cities]
        print(f"Number of log_ids in {split_name}: {len(log_ids_in_split)}")

        # Save list of log_ids in split as txt file
        with open(f"{out_folder}/{split_name}.txt", "w") as f:
            for log_id in log_ids_in_split:
                f.write(f"{log_id}\n")
        


        


    


from utils import *
import argparse
from conversions.convert_maptr import convert_maptr
from conversions.convert_maptrv2 import convert_maptrv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="Method to use to generate the splits", default="maptrv2",
                        choices=["maptr", "maptrv2"])
    parser.add_argument("--pkl_dir", type=str, help="Path to the folder of the original pickle files")
    parser.add_argument("--split_name", type=str, help="Name of the split", default="argo_geo")
    parser.add_argument("--out_dir", help="Path to the folder of the output pickle files", default=None)
    parser.add_argument("--og_pkl_name",type=str, help="Name of the original pickle files", default="argoverse_map_infos_temporal")
    args = parser.parse_args()

    method = args.method
    pkl_dir = args.pkl_dir
    split_name = args.split_name
    out_dir = args.out_dir
    og_pkl_name = args.og_pkl_name

    if out_dir is None:
        out_dir = pkl_dir

    print(f"Generating {split_name} split for {method}")
    new_train_pkl_name = f"{out_dir}/{method}_{split_name}_train.pkl"
    new_val_pkl_name = f"{out_dir}/{method}_{split_name}_val.pkl"
    new_test_pkl_name = f"{out_dir}/{method}_{split_name}_test.pkl"

    # Test if the original pickle files exist
    train_pkl_name = f"{og_pkl_name}_train.pkl"
    train_pkl_path = os.path.join(pkl_dir, train_pkl_name)
    print(f"Will attempt to load pickles with basename: ", og_pkl_name)
    if not os.path.exists(train_pkl_path):
        print(f"Did not find {train_pkl_path}. Use --og_pkl_name to specify the base name of your original pkl files")
        raise ValueError(f"File not found: {train_pkl_path}")
    
    # Load split name
    OUTPUT_ROOT = f"{SPLIT_ROOT}/{split_name}"
    split_pkl_file = f"{OUTPUT_ROOT}/{split_name}.pkl"
    if not os.path.exists(split_pkl_file):
        split_pkl_file = "geo_splits/argo_geo.pkl"
    print(f"Loading split: {split_name} from {split_pkl_file}")
    with open(split_pkl_file, "rb") as f:
        token_to_set = pickle.load(f)

    if method == "maptr":
        train, val, test = convert_maptr(pkl_dir, token_to_set, og_pkl_name)
    elif method == "maptrv2":
        train, val, test = convert_maptrv2(pkl_dir, token_to_set, og_pkl_name)
    else:
        raise ValueError("Invalid method: ", method)
    
    print("Saving new pickles...")
    with open(new_train_pkl_name, "wb") as f:
        pickle.dump(train, f)
    with open(new_val_pkl_name, "wb") as f:
        pickle.dump(val, f)
    with open(new_test_pkl_name, "wb") as f:
        pickle.dump(test, f)

    print("Done!")
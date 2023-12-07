from utils import *

def convert_mars(pkl_dir, token_to_set, og_pkl_name="nuscenes_map_infos_temporal"):
    train_pkl_name = f"{og_pkl_name}_train.pkl"
    val_pkl_name = f"{og_pkl_name}_val.pkl"
    test_pkl_name = f"{og_pkl_name}_test.pkl"

    train_pkl_path = os.path.join(pkl_dir, train_pkl_name)
    val_pkl_path = os.path.join(pkl_dir, val_pkl_name)
    test_pkl_path = os.path.join(pkl_dir, test_pkl_name)

    print("Loading pickles...")
    with open(train_pkl_path, "rb") as f:
        train_pkl = pickle.load(f)
    with open(val_pkl_path, "rb") as f:
        val_pkl = pickle.load(f)
    with open(test_pkl_path, "rb") as f:
        test_pkl = pickle.load(f)

    scene_to_original_pickles = {"train": train_pkl, "val": val_pkl, "test": test_pkl}

    print("Creating new splits...")
    new_train_infos = []
    new_val_infos = []
    new_test_infos = []

    for split, pkls in scene_to_original_pickles.items():
        for pkl in pkls:
            token = pkl["token"]

            if token not in token_to_set:
                continue

            new_split_name = token_to_set[token]

            if new_split_name == "train":
                new_train_infos.append(pkl)
            elif new_split_name == "val":
                new_val_infos.append(pkl)
            elif new_split_name == "test":
                new_test_infos.append(pkl)
            elif new_split_name == "removed":
                continue
            else:
                raise ValueError("Invalid split: ", new_split_name)
            
    new_train_pkl = new_train_infos
    new_val_pkl = new_val_infos
    new_test_pkl = new_test_infos

    return new_train_pkl, new_val_pkl, new_test_pkl
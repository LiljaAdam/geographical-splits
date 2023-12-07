from utils import *

def convert_maptr(pkl_dir, token_to_set, og_pkl_name="nuscenes_map_infos_temporal"):
    train_pkl_name = f"{og_pkl_name}_train.pkl"
    val_pkl_name = f"{og_pkl_name}_val.pkl"
    test_pkl_name = f"{og_pkl_name}_test.pkl"

    train_pkl_path = os.path.join(pkl_dir, train_pkl_name)
    val_pkl_path = os.path.join(pkl_dir, val_pkl_name)
    test_pkl_path = os.path.join(pkl_dir, test_pkl_name)
    
    with open(train_pkl_path, "rb") as f:
        train_pkl = pickle.load(f)
    with open(val_pkl_path, "rb") as f:
        val_pkl = pickle.load(f)
    with open(test_pkl_path, "rb") as f:
        test_pkl = pickle.load(f)

    scene_to_original_pickles = {"train": train_pkl, "val": val_pkl, "test": test_pkl}

    new_train_infos = []
    new_val_infos = []
    new_test_infos = []

    num_removed = 0
    print("Creating new splits...")
    for split, pkls in scene_to_original_pickles.items():
        for pkl in pkls["infos"]:
            token = pkl["token"]
            new_split = token_to_set[token]
            if not "valid_flag" in pkl:
                pkl["valid_flag"] = [None]
            if new_split == "train":
                new_train_infos.append(pkl)
            elif new_split == "val":
                new_val_infos.append(pkl)
            elif new_split == "test":
                new_test_infos.append(pkl)
            elif new_split == "removed":
                num_removed += 1
                continue
            else:
                raise ValueError("Invalid split: ", new_split)

    new_train_pkl = {"infos": new_train_infos, "metadata": train_pkl["metadata"]}
    new_val_pkl = {"infos": new_val_infos, "metadata": val_pkl["metadata"]}
    new_test_pkl = {"infos": new_test_infos, "metadata": test_pkl["metadata"]}

    return new_train_pkl, new_val_pkl, new_test_pkl
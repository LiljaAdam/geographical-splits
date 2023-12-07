from utils import *

def convert_maptr(pkl_dir, token_to_set, og_pkl_name="nuscenes_map_infos_temporal"):
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

    new_pkl = {
               "train": {"samples": [], "id2map": {}},
               "val": {"samples": [], "id2map": {}},
               "test": {"samples": [], "id2map": {}}
            }

    print("Creating new splits...")
    for split, pkls in scene_to_original_pickles.items():
        for sample in pkls["samples"]:
            sample_log_id = sample["log_id"]
            id2map = pkls["id2map"][sample_log_id]
            new_split = token_to_set[sample_log_id]

            new_pkl[new_split]["samples"].append(sample)
            new_pkl[new_split]["id2map"][sample_log_id] = id2map

    return new_pkl["train"], new_pkl["val"], new_pkl["test"]
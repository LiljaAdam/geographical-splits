# Geographical Splits for nuScenes and Argoverse 2

## Text files
In the `*_txts`folders the set is indicated by the file name. E.g. `train.txt` contains the scenes/samples that are used for training. 

## Pickle files
The pickle files contain the geographical splits for nuScenes and Argoverse 2 and can be loaded through:
```
import pickle
with open('nusc_geo.pkl', 'rb') as f:
    nuscenes_geo_splits = pickle.load(f)
```

For nuScenes the content is a dictionary where each sample token is mapped to a set (train, val, or test).

For Argoverse the content is a dictionary where each log id is mapped to a set (train, val, or test).

# Geographical Splits for nuScenes and Argoverse 2
The two pickle files contain the geographical splits for nuScenes and Argoverse 2. 
They can be loaded through:
```
import pickle
with open('nusc_geo.pkl', 'rb') as f:
    nuscenes_geo_splits = pickle.load(f)
```

For nuScenes you will get a dictionary where each sample token is mapped to a set (train, val, or test).

For Argoverse you will get a dictionary where each log id is mapped to a set (train, val, or test).

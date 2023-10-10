# Prepare nuScenes 3D Detection and 3D Occupancy Data
**1. Download nuScenes V1.0 full dataset and can bus data [HERE](https://www.nuscenes.org/download). Organize the folder structure:**
```
OccNet
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
```

**2. Prepared nuScenes 3D detection data**
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```
Using the above code will generate the following files
`
data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl
`

**3. Prepare 3D Occupancy dataset**

We provide the 3D occupancy and flow annotations for the train and validation dataset.
The annotations is defined in the LiDAR cooridinate system including 16 classes. 
> Dataset Information
<div align="left">
  
| Type |  Info |
| :----: | :----: |
| train \| val          | 28130 \| 6019 |
| annotations             | occupancy \|  flow |
| cameras         | 6 |
| range           | [-50m, -50m, -5m, 50m, 50m, 3m]|
| volume size     | [200, 200, 16]|
| #classes        | 0 - 15 |
</div>


1.Dowload train and validataion dataset and put it in the `data` folder

| Version | voxel size | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: | :---: |
| occ_gt_release_v1_0  |  0.5m| [train_val](https://drive.google.com/file/d/1Ds7NY475sS13A9KErr-MHlOBEY1oFi76/view?usp=sharing) | [train_val](https://pan.baidu.com/s/1O4iCdY7DOWts9KAIuRNT2A?pwd=hgk2) | ~15G |


2.unzip the file
```
tar -zxvf occ_gt_release_v1_0.tar.gz
```
You will obtain the folder structure
```
OccNet
├── data/
│   ├── occ_gt_release_v1_0/
│   │   ├── train/
│   │   ├── val/
│   │   ├── occ_gt_train.json
│   │   ├── occ_gt_val.json
```

3.Merge 3D detection and 3D occupancy dataset
```
python tools/create_data_with_occ.py
```
Using the above code will generate the following files
`
data/occ_gt_release_v1_0/nuscenes_infos_temporal_{train,val}_occ_gt.pkl
`

We also provide the downlink of theses pkls.
| Version | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | 
| :---: | :---: | :---: | 
| occ_gt_release_v1_0  | [train](https://drive.google.com/file/d/1iaJk40ieqoYDd_VjZALDbnRJHGpQ3Ybx/view?usp=sharing) \| [val](https://drive.google.com/file/d/1lE9h8t5dFVdZ9dBg01jTg7GeiAWIeytZ/view?usp=sharing) | [train](https://pan.baidu.com/s/1vzFGs6g9g7f_08QrItfVGw?pwd=djsh) \| [val](https://pan.baidu.com/s/1flOglbPh5BDb0i8QfpcIbQ?pwd=ntys) | 


4.The data structure of the project is organized as:
```
OccNet
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl   
│   ├── occ_gt_release_v1_0/
│   │   ├── train/
│   │   ├── val/
│   │   ├── occ_gt_train.json
│   │   ├── occ_gt_val.json
│   │   ├── nuscenes_infos_temporal_train_occ_gt.pkl
│   │   ├── nuscenes_infos_temporal_val_occ_gt.pkl
```

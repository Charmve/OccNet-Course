# Dataset Stats

<p align="center">
  <img src="../assets/OpenScene_data_stats.gif" width="996px" >
</p>





## The Largest Up-to-Date Dataset in Autonomous Driving
Comparison to prevailing benchmarks in the wild: 


|  Dataset  |      Sensor Data (hr)     | Scan | Annotated Fame |  Sensor Setup | Annotation | Ecosystem |
|:---------:|:--------------------:|:---------:|:-------------:|:------:|:--------------------------------------------:|:----------------:|
| [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)  |           1.5  |  15K | 15K         | 1L 2C    | 3D box, segmentation, depth, flow | [Leaderboard](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) |
| [Waymo](https://waymo.com/open/)   |             6.4  |  230K | 230K   | 5L 5C    | 3D box, flow  | [Challenge](https://waymo.com/open/challenges/) |
| [nuScenes](https://www.nuscenes.org/)   |             5.5  |  390K | 40K  | 1L 6C  | 3D box, segmentation  | [Leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) |
| [Lyft](https://self-driving.lyft.com/level5/data/) | 2.5|   323K | 46K | 3L 7C | 3D box | - |
| [ONCE](https://once-for-auto-driving.github.io/)   |            144  |  1M | 15K | 1L 7C  | 3D box, 3D lane  | - |
| [BDD100k](https://www.vis.xyz/bdd100k/)   |            1000  |  100K | 100K| 1C  | 2D box, 2D lane :cry:  | [Workshop](https://www.vis.xyz/bdd100k/challenges/cvpr2023/) |
| **OpenScene** |          **:boom: 120**  |  **:boom: 40M** |  **:boom: 4M** | 5L 8C  | Occupancy :smile: | [Leaderboard](https://opendrivelab.com/AD23Challenge.html#Track3) <br> [Challenge](https://opendrivelab.com/AD24Challenge.html) <br> [Workshop](https://opendrivelab.com/e2ead/cvpr23.html) |

> L: LiDAR, C: Camera


## Fact Sheet

<center>

|  Type  | Info | 
|:---------:|:-----------------|
| Location | Las Vegas (64%), Singapore (15%), Pittsburgh (12%), Boston (9%) |
| Duration | 1521 logs, 120+ hours |
| Scenario category | Dynamics: 5 types (e.g. high lateral acceleration) <br>  Interaction: 18 types (e.g. waiting for pedestrians to cross) <br> Zone: 8 types (e.g. on pickup-dropoff area) <br> Maneuver: 22 types (e.g. unprotected cross turn) <br>  Behavior: 22 types (e.g. stopping at a traffic light with a lead vehicle ahead) |
| Track| Frequency of tracks/ego: 2hz <br> Average length of scenes: 20s |
| Class| Vehicle, Bicycle, Pedestrian, Traffic cone, Barrier, Construction zone sign, Generic object, Background |
| Split | trainval (1310 logs), test (147 logs), mini (64 logs) |
| Voxel | Range: [-50m, -50m, -4m, 50m, 50m, 4m]; Size: 0.5m |
<!---| Scenarios |  Total unique scenario types |--->

</center>

## Filesystem Hierarchy
The final hierarchy should look as follows (depending on the splits downloaded above):
```angular2html
~/OpenScene
├── assets
├── docs
├── DriveEngine
│   └── ${USER CODE}
│       ├── project
│       │   └── my_project
│       └── exp
│           └── my_openscene_experiment
└── dataset
    ├── openscene-v1.0 (optional)
    |   ├── occupancy (optional)
    └── openscene-v1.1
        ├── meta_datas
        |     ├── mini
        │     │     ├── 2021.05.12.22.00.38_veh-35_01008_01518.pkl
        │     │     ├── 2021.05.12.22.28.35_veh-35_00620_01164.pkl
        │     │     ├── ...
        │     │     └── 2021.10.11.08.31.07_veh-50_01750_01948.pkl
        |     ├── trainval
        |     └── test
        |     
        └── sensor_blobs
              ├── mini
              │    ├── 2021.05.12.22.00.38_veh-35_01008_01518                                           
              │    │    ├── CAM_F0
              │    │    │     ├── c082c104b7ac5a71.jpg
              │    │    │     ├── af380db4b4ca5d63.jpg
              │    │    │     ├── ...
              │    │    │     └── 2270fccfb44858b3.jpg
              │    │    ├── CAM_B0
              │    │    ├── CAM_L0
              │    │    ├── CAM_L1
              │    │    ├── CAM_L2
              │    │    ├── CAM_R0
              │    │    ├── CAM_R1
              │    │    ├── CAM_R2
              │    │    └── MergedPointCloud
              │    │            ├── 0079e06969ed5625.pcd
              │    │            ├── 01817973fa0957d5.pcd
              │    │            ├── ...
              │    │            └── fffb7c8e89cd54a5.pcd       
              │    ├── 2021.06.09.17.23.18_veh-38_00773_01140 
              │    ├── ...                                                                            
              │    └── 2021.10.11.08.31.07_veh-50_01750_01948
              ├── trainval
              └── test

```


## Meta Data
Each `.pkl` file is stored in the following format：

```
{
    'token':                                <str> -- Unique record identifier. Pointing to the nuPlan lidar_pc.
    'frame_idx':                            <int> -- Indicates the idx of the current frame.
    'timestamp':                            <int> -- Unix time stamp.
    'log_name':                             <str> -- Short string identifier.
    'log_token':                            <str> -- Foreign key pointing to the nuPlan log.
    'scene_name':                           <str> -- Short string identifier.
    'scene_token':                          <str> -- Foreign key pointing to the nuPlan scene.
    'map_location':                         <str> -- Area where log was captured.
    'roadblock_ids':                        <list> -- A sequence of roadblock ids separated by commas. The ids can be looked up in the nuPlan Map API.
    'vehicle_name':                         <str> -- String identifier for the current car.
    'can_bus':                              <list> -- Used for vehicle communications, including low-level information about position, speed, acceleration, steering, lights, batteries, etc.
    'lidar_path':                           <str> -- The relative address to store the lidar data.
    'lidar2ego_translation':                <list> -- Translation matrix from the lidar coordinate system to the ego coordinate system.
    'lidar2ego_rotation':                   <list> -- Rotation matrix from the lidar coordinate system to the ego coordinate system.
    'ego2global_translation':               <list> -- Translation matrix from ego coordinate system to global coordinate system
    'ego2global_rotation':                  <list> -- Rotation matrix from ego coordinate system to global coordinate system
    'ego_dynamic_state':                    <list> -- The velocity and acceleration of ego car.
    'traffic_lights':                       <list> -- The status of traffic lights.
    'driving_command':                      <list> -- The high-level driving command. One-hot with 4-classes: (left, forward, right, unknown).
    'cams': {
        'CAM_F0': {
            'data_path':                    <str> -- The relative address to store the camera_front_0 data.
            'sensor2lidar_rotation':        <list> -- Rotation matrix from camera_front_0 sensor to lidar coordinate system.
            'sensor2lidar_translation':     <list> -- Translation matrix from camera_front_0 sensor to lidar coordinate system.
            'cam_intrinsic':                <list> -- Intrinsic matrix of the camera 
            'distortion':                   <list> -- The distortion coefficiants of camera.
        }
        'CAM_L0':                           <dict> -- Camera configuration.
        'CAM_R0':                           <dict> -- Camera configuration.
        'CAM_L1':                           <dict> -- Camera configuration.
        'CAM_R1':                           <dict> -- Camera configuration.
        'CAM_L2':                           <dict> -- Camera configuration.
        'CAM_R2':                           <dict> -- Camera configuration.
        'CAM_B0':                           <dict> -- Camera configuration.
    }
    'sample_prev':                          <str> -- Foreign key. Sample that precedes this in time. Empty if start of scene.
    'sample_next':                          <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
    'ego2global':                           <list> -- Ego to the global coordinate system transformation matrix.
    'lidar2ego':                            <list> -- Lidar to the ego coordinate system transformation matrix.
    'lidar2global':                         <list> -- Lidar to the global coordinate system transformation matrix.
    'anns': {
        'gt_boxes':                         <list> -- Ground truth boxes. (x,y,z,l,w,h,yaw)
        'gt_names':                         <list> -- Class names.
        'gt_velocity_3d':                   <list> -- 3D velocity.
        'instance_tokens':                  <list> -- Unique record identifier of single frame instance.
        'track_tokens':                     <list> -- Unique record identifier of tracking instance.
    }
    'occ_gt_final_path':                    <str> -- The relative address to store the occupancy gt data.
    'flow_gt_final_path':                   <str> -- The relative address to store the flow gt data.     
}
```

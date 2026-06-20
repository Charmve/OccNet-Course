# cuPCL

<a><img src="https://img.shields.io/badge/-Documentation-bright"/></a>

cuPCL has some libraries used to process points cloud with CUDA and some samples for their usage.
There are several subfolders in the project and every subfolder has:

1. lib implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's

To get started, follow the instructions below.

**Xavier, Orin, and Linux x86 are supported(For Jetpack 4.x, Jetpack 5.x, and Linux x86_64 library, please check the respective branch).**

If you run into any issues please [let us know](../../issues).

## Getting Started

To get started, follow these steps.

### Step 1 - Install Dependencies

Install PCL (Eigen included)

```
$sudo apt-get update
$sudo apt-get install libpcl-dev
```

### Step 2 - Build

Enter any subfolder and then

```
make
```

### Step 3 - Run

1. Please boost CPU and GPU firstly

```
sudo nvpmodel -m 0
sudo jetson_clocks
```

2. Usage:

```
./demo [*.pcd]
```

## How to check the Version of the Lib

```
$ strings lib* | grep version | grep lib
lib* version: 1.0 Jun  2 2019 09:30:19
```

## Test Enviroment

```
Jetson Xavier AGX 8GB
Jetpack 4.4.1
CUDA 10.2
PCL 1.8
Eigen 3
```

## Functions List

### cuICP

This project provides:

1. lib for Icp implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's
3. two point clounds: test_P.pcd and test_Q.pcd that both having 7000 points

### cuFilter

The project provides:<br>

1. lib for Filter implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's
3. A point clound: sample.pcd which has 119978 points

NOTE: Now it supports two kinds of filters: PassThrough and VoxelGrid.

### cuSegmentation

This package provides:<br>

1. lib for Segmentation implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's
3. A point clound: sample.pcd which has 119978 points

NOTE: Now it just supports SAC_RANSAC + SACMODEL_PLANE.

### cuOctree

This package provides:<br>

1. lib for Octree implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's
3. A point clound: sample.pcd which has 119978 points

NOTE: Now it just supports Radius Search and Approx Nearest Search

### cuCluster

This package provides:<br>

1. lib for Cluster implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's

NOTE:

1. Cluster can be used to extract objects from points cloud after road plane was removed by Segmentation.
2. The sample will use a PCD(sample.pcd) file which had removed road plane.

### cuNDT

This package provides:

1. lib for NDT implemented by CUDA
2. Sample code showing the lib usage and checking the perf and accuracy by comparing its output with PCL's
3. two point clounds: test_P.pcd and test_Q.pcd that both having 7000 points

## Performance Comparison

### cuICP

||GPU|CPU-GICP|CPU-ICP|
|---|---|----|---|
|count of points cloud|7000|7000|7000|
|maximum of iterations|20|20|20|
|cost time(ms)|43.3|652.8|7746.0|
|fitness_score(the lower the better)|0.514|0.525|0.643|

### cuFilter

#### Pass Through

||GPU|CPU|
|-|-|-|
|count of points cloud|11w+|11w+|
|down,up FilterLimits|(-0.5, 0.5)|(-0.5, 0.5)|
|limitsNegative|false|false|
|Points selected|5110|5110|
|cost time(ms)|0.660954|2.97487|

#### VoxelGrid

||GPU|CPU|
|-|-|-|
|count of points cloud|11w+|11w+|
|LeafSize|(1,1,1)|(1,1,1)|
|Points selected|3440|3440|
|cost time(ms)|3.12895|7.26262|

### cuSegmentation

||GPU|CPU|
|-|-|-|
|segment by time(ms)|14.9346|69.6264|
|model coefficients|{-0.00273056, 0.0425288, 0.999092, 1.75528}|{-0.00273045, 0.0425287, 0.999092, 1.75528}|
|find points|9054|9054|

### cuOctree

||GPU|CPU|
|-|-|-|
|count of points cloud|119978|119978|
|down,up FilterLimits|(0.0,1.0)|(0.0,1.0)|
|limitsNegative|false|false|
|Points selected|16265|16265|
|cost time(ms)|0.589752|2.82811|

### cuCluster

||GPU|CPU|
|-|-|-|
|Count of points cloud|17w+|17w+|
|Cluster cost time(ms)|10.3122|4016.85|

### cuNDT

||GPU|CPU|
|-|-|-|
|count of points cloud|7000|7000|
|cost time(ms)|34.7789|136.858|
|fitness_score(the lower the better)|0.538|0.540|

## Official Blog

<https://developer.nvidia.com/blog/accelerating-lidar-for-robotics-with-cuda-based-pcl/>
<https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/>

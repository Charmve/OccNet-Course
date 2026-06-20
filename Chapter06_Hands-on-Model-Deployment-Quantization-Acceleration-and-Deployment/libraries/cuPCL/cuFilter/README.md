## Description
The are two folders in the package
1. lib for segmentation implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL
3. A point clound: sample.pcd which has 119978 points

## Prerequisites

### Install PCL (Eigen included)
```
$sudo apt-get update
$sudo apt-get install libpcl-dev
```
## Build
$ make

## Run
Please boost CPU and GPU firstly

```
sudo nvpmodel -m 0
sudo jetson_clocks 
```
Usage:<br>
```
./demo [*.pcd]
```
## How to check output
We can get output as below:
```
------------checking CUDA ---------------- 
CUDA Loaded 119978 data points from PCD file with the following fields: x y z

------------checking CUDA PassThrough ---------------- 
CUDA PassThrough by Time: 0.660954 ms.
CUDA PassThrough before filtering: 119978
CUDA PassThrough after filtering: 5110

------------checking CUDA VoxelGrid---------------- 
CUDA VoxelGrid by Time: 3.12895 ms.
CUDA VoxelGrid before filtering: 119978
CUDA VoxelGrid after filtering: 3440


------------checking PCL ---------------- 
PCL(CPU) Loaded 119978 data points from PCD file with the following fields: x y z

------------checking PCL(CPU) PassThrough ---------------- 
PCL(CPU) PassThrough by Time: 2.97487 ms.
PointCloud before filtering: 119978 data points (x y z).
PointCloud after filtering: 5110 data points (x y z).

------------checking PCL VoxelGrid---------------- 
PCL VoxelGrid by Time: 7.26262 ms.
PointCloud before filtering: 119978 data points (x y z).
PointCloud after filtering: 3440 data points (x y z).

```


**Perforamnce table**
-------------------------------------------------
```
PASSTHROUGH             GPU             CPU
count of points cloud   11w+            11w+
dim                     X               X
down,up FilterLimits    (-0.5, 0.5)     (-0.5, 0.5)
limitsNegative          false           false
Points selected         5110            5110
cost time(ms)           0.660954        2.97487
```
```
VOXELGRID               GPU             CPU
count of points cloud   11w+            11w+
LeafSize                (1,1,1)         (1,1,1)
Points selected         3440            3440
cost time(ms)           3.12895         7.26262
```
**How To Check the Version of the Lib**
-------------------------------------------------
```
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>
```
**Test Enviroment**
-------------------------------------------------
Jetson Xavier AGX 8GB<br>
Jetpack 4.4.1<br>
CUDA 10.2<br>
PCL 1.8<br>
Eigen 3<br>


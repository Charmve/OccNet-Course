## Description
This package provides:<br>
1. Cluster lib implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL

NOTE:
Cluster can be used to extract objects from points cloud after road plane was removed by Segmentation.
The sample will use a PCD(sample.pcd) file which had been removed road plane.

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
-------------- test CUDA lib -----------
-------------- cudaExtractCluster -----------
CUDA extract by Time: 10.3122 ms.
PointCloud representing the Cluster: 162152 data points.
PointCloud representing the Cluster: 7098 data points.
PointCloud representing the Cluster: 1263 data points.
PointCloud representing the Cluster: 257 data points.

-------------- test PCL lib -----------
PCL(CPU) cluster kd-tree by Time: 60.8192 ms.
PCL(CPU) cluster extracted by Time: 4016.85 ms.
PointCloud cluster_indices: 4.
PointCloud representing the Cluster: 166789 data points.
PointCloud representing the Cluster: 7410 data points.
PointCloud representing the Cluster: 1318 data points.
PointCloud representing the Cluster: 427 data points.

```
## Perforamnce table
```
                                GPU         CPU
Count of points cloud           17w+        17w+
Cluster cost time(ms)           10.3122     4016.85
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



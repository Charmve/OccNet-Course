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
./demo src.pcd  dst.pcd
```
## How to check output
We can get output as below:
```
------------checking CUDA Approx nearest search---------------- 
CUDA costs : 2.55582 ms.
Point distance AVG: 0.721553

------------checking CUDA radiusSearch---------------- 
CUDA costs: 0.083172 ms.
Points selected: 4751

------------checking OC-Tree creating ---------------- 
PCL(CPU) create oc-tree by Time: 4.35747 ms.

------------checking PCL(CPU)  Approx nearest search ---------------- 
PCL(CPU) costs: 11.6701 ms.
Point distance AVG: 2.75023

------------checking PCL(CPU) radiusSearch ---------------- 
PCL(CPU) costs by Time: 1.29362 ms.
Points selected: 4751
```


**Perforamnce table**
-------------------------------------------------
```
Approx nearest          GPU             CPU
points count of tree    7000            7000
points count of target  7000            7000
Distance Error          0.721           2.75
cost time(ms)           2.55            11.67
```
```
radiusSearch            GPU             CPU
points count of tree    7000            7000
points selected         4751            4751
cost time(ms)           0.083           1.29
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


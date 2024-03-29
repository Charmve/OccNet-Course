## Description
This package provides:<br>
1. Segmentation lib implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL
3. A point clound: sample.pcd which has 119978 points
NOTE:
Now Just support: SAC_RANSAC + SACMODEL_PLANE

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
-------------------------
CUDA segment by Time: 14.5712 ms.
CUDA modelCoefficients: -0.00269913 0.0424975 0.999093 2.10639
CUDA find points: 7519
-------------------------
PCL(CPU) segment by Time: 67.2766 ms.
Model coefficients: -0.0026991 0.0424981 0.999093 2.10639
Model inliers: 7519
```
## Perforamnce table
```
                        GPU         CPU
count of points cloud   11w+        11w+
Points selected         7519        7519
cost time(ms)           14.5712     67.2766
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



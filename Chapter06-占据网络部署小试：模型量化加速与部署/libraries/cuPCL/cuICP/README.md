## Description
This project provides:
1. lib implement ICP by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL
3. two point clound: test_P.pcd and test_Q.pcd both have 7000 points

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
./demo
```
## How to check output
We can get output as below:
```
------------checking CUDA ICP---------------- 
CUDA ICP by Time: 55.0188 ms.
CUDA ICP fitness_score: 0.514642
matrix_icp calculated Matrix by Class ICP 
Rotation matrix :
    | 0.998693 0.015291 0.048766 | 
R = | -0.013452 0.999195 -0.037806 | 
    | -0.049304 0.037101 0.998095 | 
Translation vector :
t = < 0.077319, 0.044569, 0.099613 >

------------checking PCL ICP(CPU)---------------- 
PCL icp.align Time: 444.441 ms.
has converged: 1 score: 0.525366
CUDA ICP fitness_score: 0.525366
transformation_matrix:
  0.998899  0.0107164  0.0457246  0.0790455
-0.0095028   0.999602 -0.0266788  0.0254029
-0.0459921  0.0262148   0.998599  0.0677747
         0          0          0          1

------------checking PCL GICP(CPU)---------------- 
PCL Gicp.align Time: 337.694 ms.
has converged: 1 score: 0.644919
transformation_matrix:
  0.997003  0.0286008  0.0718879  0.0629225
-0.0236729   0.997371 -0.0684918   0.242391
-0.0736578  0.0665847   0.995058   0.366667
         0          0          0          1

```

"Time" is the time ICP costed.<br>
"fitness_score" is the score of the ICP transform, the less the better.<br>

**How To Check the Version of the Lib**
-------------------------------------------------
```
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>
```
**Perforamnce table for default parameters**
-------------------------------------------------
```
                            GPU     GICP    ICP
count of points cloud       7000    7000    7000
maximum of iterations       20      20      20
cost time(ms)               55.1    364.2   523.1
fitness_score               0.514   0.644   0.525
```
**Test Enviroment**
-------------------------------------------------
Jetson Xavier AGX 8GB<br>
Jetpack 4.4.1<br>
CUDA 10.2<br>
PCL 1.8<br>
Eigen 3<br>

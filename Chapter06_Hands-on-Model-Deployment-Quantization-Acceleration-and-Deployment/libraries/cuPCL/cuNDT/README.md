**Summary**
-------------------------------------------------
This project provides:<br>
1. Lib implement by CUDA
2. Sample code can show the function usage, which can also be used to check perf
   and output verification by comparing its output from PCL
3. Generate the customer release package by one command - "$ make release"

**Source Code**
-------------------------------------------------
*  lib: the implementation of the convolution function.<br>
  More details about the API canbe found in the header file.<br>
*  sample: the sample code about how to use the function<br>
  and perf test<br>

**CUDA LIB Dependency**
-------------------------------------------------
Please install cuda toolkit, EIGEN<br>

**How to Compile**
-------------------------------------------------
The sample depends PCL to load data and show result, please install PCL firstly<br>
>./$ make<br>


**How to Run the Sample**
-------------------------------------------------
Usage:<br>
>./$(App) <br>

**How to check output**
-------------------------------------------------
We can get output like below:<br>
```
CUDA NDT by Time: 38.0663 ms.
CUDA NDT fitness_score: 0.538532
Rotation matrix :
    | 0.999171 0.010360 0.039369 | 
R = | -0.009026 0.999384 -0.033907 | 
    | -0.039696 0.033523 0.998649 | 
Translation vector :
t = < 0.056862, 0.143134, 0.188664 >
```

"Time" is the time CUDA costed.<br>
"fitness_score" is the score of the CUDA transform, the less the better.<br>


**How To Check the Version of the Lib**
-------------------------------------------------
```
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>

```
**Perforamnce table for default parameters**
-------------------------------------------------
```
                        GPU     CPU
count of points cloud   7000    7000
maximum of iterations 	35 	    35
resolution              1.0     1.0
cost time(ms) 	        38      153
fitness_score           0.538   0.540
```
**Test Enviroment**
-------------------------------------------------
Xavier<br>
jetpack 4.4.1<br>
cuda 10.2<br>
Eigen 3<br>
PCL 1.8<br>

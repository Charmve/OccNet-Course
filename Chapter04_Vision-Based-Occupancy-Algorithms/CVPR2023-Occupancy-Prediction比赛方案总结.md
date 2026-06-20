

https://zhuanlan.zhihu.com/p/638481909

3D Occupancy Prediction（Occ）是Telsa在2022 AI Day里提出的检测任务，任务的提出是认为此前的3D目标检测所检测出的3D目标框，不足描述一般物体（数据集中没有的物体），在此任务中，则把物体切分成体素进行表达，要求网络可以在3D体素空间中，预测每个体素的类别，可以认为是语义分割在3D体素空间的扩展任务，具体预测图如下图所示。

<p align="center">
    <img title="occupanc" src="../src/imgs/occupanc_1.gif">
    <br>From BEV to Occupancy Network
    <br><sup>*From https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction</sup>
</p>

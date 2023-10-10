## Small scale chopper

将每一帧的bev矢量图拼接起来

## Final Project

复现一个纯视觉方案，走通数据标定、3D目标检测、BEV视角坐标对齐、3D语义场景补全补全完整流程。基于[BEVFormer](https://github.com/fundamentalvision/BEVFormer)，结合BEVDepth，完成占据栅格的预测。

- 基础：基于 bevformer_base 完成占据栅格的预测，给出可视化结果；

- 进阶：基于 bevformer_small 完成占据栅格的预测，并在英伟达 Drive Orin 上通过TensorRT部署，CUDA加速；


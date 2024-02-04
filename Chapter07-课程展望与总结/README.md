<!--
 * @Author: Charmve yidazhang1@gmail.com
 * @Date: 2023-10-10 10:49:13
 * @LastEditors: Charmve yidazhang1@gmail.com
 * @LastEditTime: 2024-02-02 01:16:50
 * @FilePath: /OccNet-Course/Chapter07-课程展望与总结/README.md
 * @Version: 1.0.1
 * @Blogs: charmve.blog.csdn.net
 * @GitHub: https://github.com/Charmve
 * @Description: 
 * 
 * Copyright (c) 2023 by Charmve, All Rights Reserved. 
 * Licensed under the MIT License.
-->


在本专题课程的课程展望和总结中，主要从算法框架、数据、仿真和其他四个方面做未来展望，以及对本课程做一个总结。

- <b>算法框架</b>
    - 数据驱动的端到端 [UniAD](https://github.com/OpenDriveLab/UniAD)
      - https://mp.weixin.qq.com/s/qcNtRsBD5aadkavU9TfpFA
      - https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving
            - End-to-end Interpretable Neural Motion Planner [paper](https://arxiv.org/abs/2101.06679)
            - End-to-End Learning of Driving Models with Surround-View Cameras and Route Planners [paper](https://arxiv.org/abs/1803.10158)
      - https://github.com/E2E-AD/AD-MLP
      - ST-P3 [paper](https://arxiv.org/abs/2207.07601) | [code](https://github.com/OpenDriveLab/ST-P3)
      - MP3 [paper](https://arxiv.org/abs/2101.06806) | [video](https://www.bilibili.com/video/BV1tQ4y1k7BX)
      - TCP [NeurIPS 2022] Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline. [paper](https://arxiv.org/abs/2206.08129) | [video](https://www.bilibili.com/video/BV1Pe4y1x7E3/?spm_id_from=333.337.search-card.all.click&vd_source=57394ba751fad8e6886be567cccfa5bb) ｜[code](https://github.com/OpenDriveLab/TCP)
      - 鉴智机器人 GraphAD 
      - 
    - 大模型 [LMDrive](https://github.com/opendilab/LMDrive) [关于大模型和自动驾驶的几个迷思](关于大模型和自动驾驶的几个迷思.md)
    - 世界模型：Drive-WM、DriveDreamer
    - 矢量地图在线建图：MapTRv2、ScalableMap、VectorMapNet、HDMapNet、GeMap、MapEX
    - BEV-OCC-Transformer: OccFormer、OccWorld、Occupancy Flow

- <b>数据</b>
    - 4D数据自动标注: 
      - OCC与Nerf联合标注
      - [面向BEV感知的4D标注方案](https://zhuanlan.zhihu.com/p/642735557?utm_psn=1706841959639998464)
    - 数据合成：DrivingDiffusion、[MagicDrive](https://zhuanlan.zhihu.com/p/675303127)、UrbanSyn
    - https://github.com/runnanchen/CLIP2Scene

- <b>仿真</b>
    - [UniSim](https://waabi.ai/unisim/)
  - DRIVE Sim

- <b>其他</b>
    - 舱驾一体
    - AI 编译器: MLIR、TVM、XLA、Triton
    - 模型剪枝、模型蒸馏、模型压缩、模型量化（PTQ、QAT）


关注科技前沿公司：[Waabi](https://waabi.ai/unisim/)、[Wayve](https://wayve.ai/)
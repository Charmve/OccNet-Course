# Development environment setup

## Step 0: Install Ubuntu System:
if needed build USB with iSO, use this [ubuntu18.04.5](https://releases.ubuntu.com/18.04/ubuntu-18.04.5-desktop-amd64.iso.torrent?_ga=2.134206414.1776276798.1599202199-1172990662.1594698333)

## Step 1: Install Nvidia driver
Install GPU driver and docker on ubunutu-18.04 here:
```
# If it is thinkPad T490 & Huawei Mate 2020, disable Secure Boot in the BIOS first

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo reboot

# The system kernel may change after the update. You need to restart the machine to ensure that the driver is installed on the new kernel
---
# List available drivers
sudo ubuntu-drivers devices
sudo apt install -y nvidia-driver-470
sudo reboot
nvidia-smi 
```
# If the driver is displayed, the installation is successful
If you cannot install it，Following this link on installing nvidia driver: https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux


## Step 2: Install Docker && Nvidia-Docker2
```
# Install Docker
sudo apt install -y docker.io
sudo usermod -a -G docker $USER
---
# Install Nvidia-Docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker  
sudo reboot  
```

*The environment is available if the following information appears

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu18.04 nvidia-smi
---       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 206...  Off  | 00000000:09:00.0  On |                  N/A |
| 44%   34C    P8    11W / 184W |    110MiB /  7979MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

If you cannot install it, follow the instruction here to install nvidia-docker2：
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian

## Step 3: Get Code from GitHub
```
git clone https://github.com/Charmve/OccNet-Course

```


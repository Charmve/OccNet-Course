FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# apt换源，安装pip
RUN echo "==> 换成清华源，并更新..."  && \
    sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list  && \
    sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list  && \
    apt-get clean  && \
    apt-get update

RUN apt-get update && apt-get install wget tree vim curl jq shellcheck -yq
RUN apt-get install build-essential g++ gcc cmake -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install -U pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install Pillow==8.4.0 tqdm torchpack nuscenes-devkit mpi4py==3.0.3 numba==0.48.0
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0

ENV LANG C.UTF-8
ENV PS1 "[\[\e[1;32m\]\u\[\e[m\]\[\e[1;33m\]@\[\e[m\]\[\e[1;35m\]\h\[\e[m\]:\[\e[0;32m\]\w\[\e[0m\]$(__git_ps1 "\[\e[33m\](%s) \[\e[0m\]")\[\e[31m\]$(git_dirty)\[\e[0m\]] $ "
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER $NAME

# Install some necessary tools.
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    htop \
    locales \
    python3 \
    python3-pip \
    sudo \
    unzip \
    vim \
    git \
    wget \
    zsh \
    libssl-dev \
    libffi-dev \
&& rm -rf /var/lib/apt/lists/*

# Configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Python packages
RUN pip3 install --upgrade \
    scipy \
    numpy \
    jupyter notebook \
    torch==1.4.0 \
    torchvision==0.5.0 \
    ipdb \
    pyyaml \
    easydict

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5" && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..



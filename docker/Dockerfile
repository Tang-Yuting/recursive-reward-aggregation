FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
WORKDIR /workspace

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    bash-completion \
    build-essential \
    patchelf \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    htop \
    tmux \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
# installing mujoco distr
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco --no-same-owner \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

COPY requirements/requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN python -c "import mujoco_py"
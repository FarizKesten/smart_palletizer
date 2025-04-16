# Base ROS Noetic image
FROM ros:noetic-ros-base

# Use bash for all shell commands
SHELL ["/bin/bash", "-c"]

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS desktop tools and perception libraries
RUN apt update && apt install -y \
    ros-noetic-desktop-full \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-rviz \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-image-view \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install development tools and common libraries
RUN apt update && apt install -y \
    build-essential \
    git \
    vim \
    tmux \
    wget \
    curl \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libgtest-dev \
    libjsoncpp-dev \
    liblapack-dev \
    libopenni2-dev \
    libpcap-dev \
    libpng-dev \
    libtiff-dev \
    libusb-1.0-0-dev \
    pkg-config \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    latexmk \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# try blenderproc
# RUN apt update && apt install -y \
#     blender \
#     blenderproc \
#     && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    qemu-user-static \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (ARM-compatible Conda)
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

# Set Conda in PATH and activate for all shells
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc

# Create and activate Conda env
RUN conda init bash && \
    conda create -n rosenv python=3.8 -y && \
    echo "conda activate rosenv" >> /root/.bashrc

# Install Python packages in Conda env
RUN /opt/conda/bin/conda run -n rosenv pip install --no-cache-dir \
    open3d dash werkzeug numpy pandas scikit-learn \
    opencv-python ipympl matplotlib tqdm \
    torch torchvision ultralytics \
    notebook jupyterlab ipykernel albumentations



# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

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
    python3-pip \
    python3-setuptools \
    python3-numpy \
    python3-opencv \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    latexmk \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and IPython kernel
RUN pip3 install --no-cache-dir \
    notebook \
    jupyterlab \
    ipykernel \
    && python3 -m ipykernel install --user

# Automatically source ROS in every shell
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy and install Python requirements (e.g., for documentation)
COPY docs/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

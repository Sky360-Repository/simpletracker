# Forked from https://github.com/JulianAssmann/opencv-cuda-docker

FROM nvidia/cuda:11.4.1-devel-ubuntu20.04

ARG OPENCV_VERSION=4.5.3
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*


RUN wget \
     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && mkdir /root/.conda \
     && bash Miniconda3-latest-Linux-x86_64.sh -b \
     && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

RUN conda init bash

COPY install_ml.sh /root/install_ml.sh

RUN /root/install_ml.sh /root/miniconda3/


ARG CONDA_PREFIX=/root/miniconda3
ENV CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.9

RUN cd /opt/ &&\
     # Download and unzip OpenCV and opencv_contrib and delte zip files
     wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
     unzip $OPENCV_VERSION.zip &&\
     rm $OPENCV_VERSION.zip &&\
     wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
     unzip ${OPENCV_VERSION}.zip &&\
     rm ${OPENCV_VERSION}.zip
     # Create build folder and switch to it


# Cmake configure
RUN   mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build && \
      cmake \
         -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
         -D WITH_CUDA=ON \
         -D CMAKE_BUILD_TYPE=RELEASE \
         -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/python3.9 \
         -D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.9 \
         -D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
         -D PYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.9/site-packages \
	 -D BUILD_TIFF=ON \
	 -D BUILD_PNG=ON -D BUILD_JPEG=ON \
	 -D BUILD_ZLIB=ON \
         ..
	 

# Make
RUN cd /opt/opencv-${OPENCV_VERSION}/build && \
    make -j4

    # Install to /usr/local/lib
RUN cd /opt/opencv-${OPENCV_VERSION}/build && \
    make install && \
    ldconfig
    # &&\
    # Remove OpenCV sources and build folder
#    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


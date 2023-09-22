FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# OpenCV dependencies
RUN apt update && \
  apt upgrade -y && \
  apt install build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev -y && \
  apt install qt5-default libopencv-dev -y

RUN mkdir -p workspace/opencv_build
WORKDIR workspace/opencv_build

RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git

WORKDIR opencv
RUN git checkout 4.5.2
WORKDIR ../opencv_contrib
RUN git checkout 4.5.2

WORKDIR ../opencv 
RUN mkdir build
WORKDIR build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D WITH_QT=ON -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D BUILD_EXAMPLES=ON ..

RUN make -j8
RUN make install

RUN echo "/usr/lib/aarch64-linux-gnu" | tee --append /etc/ld.so.conf.d/opencv.conf
RUN ldconfig
RUN echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/aarch64-linux-gnu/pkgconfig" | tee --append ~/.bashrc
RUN echo "export PKG_CONFIG_PATH" | tee --append ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN source ~/.bashrc

# Not used anymore
RUN rm -r build

# Copy the application files into the image. 
WORKDIR /workspace
COPY . .

# Application dependencies
RUN apt-get update && \
  apt-get install -y python3 python3-pip wget unzip

# Download and extract LibTorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip && \
  unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip -d /workspace && \
  rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip

# Python dependencies
RUN pip3 install -r /workspace/requirements.txt

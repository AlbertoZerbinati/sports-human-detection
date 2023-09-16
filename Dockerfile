# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies and libraries for OpenCV
RUN apt update && \
    apt upgrade -y && \
    apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev \
    libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr \
    libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev \
    libdc1394-22-dev libopenexr-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev qt5-default libopencv-dev python3-pip

# Install virtualenv
RUN pip3 install virtualenv

# Create a Python virtual environment
RUN virtualenv /workspace/sports-human-detection

# Create a requirements.txt file with your Python package requirements
RUN echo "appdirs==1.4.4\n\
black==23.7.0\n\
bleach==6.0.0\n\
certifi==2023.7.22\n\
charset-normalizer==3.2.0\n\
click==8.1.7\n\
cmake==3.27.4.1\n\
contourpy==1.1.0\n\
cycler==0.11.0\n\
dlib==19.24.2\n\
docker-pycreds==0.4.0\n\
face-recognition==1.3.0\n\
face-recognition-models==0.3.0\n\
filelock==3.12.2\n\
flake8==6.1.0\n\
fonttools==4.42.1\n\
gitdb==4.0.10\n\
GitPython==3.1.34\n\
h5py==3.9.0\n\
idna==3.4\n\
imageio==2.31.2\n\
imutils==0.5.4\n\
isort==5.12.0\n\
Jinja2==3.1.2\n\
joblib==1.3.2\n\
kaggle==1.5.16\n\
kiwisolver==1.4.5\n\
lazy_loader==0.3\n\
lit==16.0.6\n\
MarkupSafe==2.1.3\n\
matplotlib==3.7.2\n\
mccabe==0.7.0\n\
mpmath==1.3.0\n\
mypy==1.5.1\n\
mypy-extensions==1.0.0\n\
networkx==3.1\n\
numpy==1.25.2\n\
nvidia-cublas-cu11==11.10.3.66\n\
nvidia-cuda-cupti-cu11==11.7.101\n\
nvidia-cuda-nvrtc-cu11==11.7.99\n\
nvidia-cuda-runtime-cu11==11.7.99\n\
nvidia-cudnn-cu11==8.5.0.96\n\
nvidia-cufft-cu11==10.9.0.58\n\
nvidia-curand-cu11==10.2.10.91\n\
nvidia-cusolver-cu11==11.4.0.1\n\
nvidia-cusparse-cu11==11.7.4.91\n\
nvidia-nccl-cu11==2.14.3\n\
nvidia-nvtx-cu11==11.7.91\n\
opencv-python==4.8.0.76\n\
packaging==23.1\n\
pandas==2.0.3\n\
pathspec==0.11.2\n\
pathtools==0.1.2\n\
Pillow==10.0.0\n\
platformdirs==3.10.0\n\
protobuf==4.24.2\n\
psutil==5.9.5\n\
pycodestyle==2.11.0\n\
pyflakes==3.1.0\n\
pyparsing==3.0.9\n\
python-dateutil==2.8.2\n\
python-slugify==8.0.1\n\
pytz==2023.3\n\
PyWavelets==1.4.1\n\
PyYAML==6.0.1\n\
requests==2.31.0\n\
scikit-image==0.21.0\n\
scikit-learn==1.3.0\n\
scipy==1.11.2\n\
sentry-sdk==1.30.0\n\
setproctitle==1.3.2\n\
six==1.16.0\n\
smmap==5.0.0\n\
sympy==1.12\n\
text-unidecode==1.3\n\
threadpoolctl==3.2.0\n\
tifffile==2023.8.25\n\
torch==2.0.1\n\
torchvision==0.15.2\n\
tqdm==4.66.1\n\
triton==2.0.0\n\
types-Pillow==10.0.0.2\n\
types-tqdm==4.66.0.2\n\
typing_extensions==4.7.1\n\
tzdata==2023.3\n\
urllib3==2.0.4\n\
wandb==0.15.9\n\
webencodings==0.5.1" > /workspace/requirements.txt

# Activate the virtual environment and install the Python packages
RUN /bin/bash -c "source /workspace/sports-human-detection/bin/activate && pip install -r /workspace/requirements.txt"

# Activate the virtual environment
ENV VIRTUAL_ENV /workspace/sports-human-detection
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

# Create workspace and clone OpenCV repos
RUN mkdir -p workspace/opencv_build 
WORKDIR workspace/opencv_build
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git

# Checkout the required version for both repos
WORKDIR opencv
RUN git checkout 4.5.2
WORKDIR ../opencv_contrib
RUN git checkout 4.5.2

# Build OpenCV
WORKDIR ../opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D WITH_QT=ON \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=ON .. && \
    make -j8 && \
    make install

# Update shared library cache
RUN echo "/usr/lib/aarch64-linux-gnu" | tee --append /etc/ld.so.conf.d/opencv.conf && \
    ldconfig && \
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/aarch64-linux-gnu/pkgconfig" | tee --append ~/.bashrc && \
    echo "export PKG_CONFIG_PATH" | tee --append ~/.bashrc

# Set the working directory
WORKDIR /workspace

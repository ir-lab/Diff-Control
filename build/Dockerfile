FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
WORKDIR /tf
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# install unix libraries
RUN apt-get update -y --fix-missing
RUN apt-get install -y ffmpeg
RUN ["apt-get", "install", "-y", "git", "libsm6", "libxext6", "libxrender-dev"]
COPY requirements.txt requirements.txt
# install pip libraries
RUN pip install --upgrade cython
RUN pip install -r requirements.txt
# run jupyter
RUN pip install jupyter

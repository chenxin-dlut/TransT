FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

# Install base packages

RUN apt-get update
RUN apt-get install -y sudo
RUN apt -y install wget


RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh


RUN bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:c:/miniconda/bin


RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        unzip \
        ca-certificates\
        libjpeg-dev \
        libpng-dev \
        software-properties-common \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log



RUN add-apt-repository universe
RUN apt-get update

# Install PyTorch
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


# Install trans-t packages


RUN apt-get install -y libturbojpeg libgl1

RUN git clone https://github.com/supervisely-ecosystem/trans-t.git /trans-t
RUN pip install -r /trans-t/requirements.txt


RUN mkdir /trans-t/data/model/
RUN cd /trans-t/data/model/ && wget --no-check-certificate 'https://github.com/supervisely-ecosystem/trans-t/releases/download/v1.0.6/transt.pth' -O transt.pth

RUN mv /trans-t/ /root/trans-t/


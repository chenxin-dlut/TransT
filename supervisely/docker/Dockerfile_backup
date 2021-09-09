FROM supervisely/base-pytorch:6.0.22


# Install PyTorch
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


# Install other packages

RUN apt-get install libturbojpeg


RUN git clone https://github.com/supervisely-ecosystem/trans-t.git
RUN pip install -r /trans-t/requirements.txt

RUN pip install supervisely==6.1.95

RUN mkdir /trans-t/data/model/
RUN cd /trans-t/data/model/ && wget --no-check-certificate 'https://drive.google.com/uc?id=1uxnvaKdTtYj-IgmrYwi2hCD5ZddTK2d8&export=download' -O transt.pth

RUN mv /trans-t/ /root/trans-t/


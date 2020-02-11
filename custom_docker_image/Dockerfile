FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN  apt-get update \ 
  && apt-get install -y wget git libsm6 libxext6 libxrender-dev libgtk2.0-dev \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update\
    && apt-get upgrade -y \
    && apt-get install bzip2 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /miniconda
RUN /bin/bash -c "source ~/.bashrc"

RUN conda init
RUN /bin/bash -c "source ~/.bashrc"

RUN conda remove -c anaconda pyyaml -y

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/ && \
 conda config --set show_channel_urls yes && \
 conda install python=3.7.1

COPY apex/ apex/
RUN pip install -v --no-cache-dir apex/

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
        && rm /tmp/requirements.txt

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple jsonschema
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple dill==0.2.8.2
RUN pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN export PYTHONPATH=/job/job_source/model1/:/job/job_source/:$PYTHONPATH
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple efficientnet-pytorch

ENTRYPOINT ["python"]
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN  apt-get update \ 
  && apt-get install -y wget git libsm6 libxext6 libxrender-dev libgtk2.0-dev \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update\
    && apt-get upgrade -y \
    && apt-get install bzip2

COPY miniconda.sh miniconda.sh
RUN bash miniconda.sh -b -p /miniconda
RUN /bin/bash -c "source ~/.bashrc" && conda init && /bin/bash -c "source ~/.bashrc"
RUN conda config --set ssl_verify no
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/ && \
 conda config --set show_channel_urls yes && conda remove pyyaml -y &&\
 conda install python=3.7.1

COPY apex/ apex/
RUN pip install -v --no-cache-dir apex/

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
        && rm /tmp/requirements.txt

RUN export PYTHONPATH=/job/job_source/:$PYTHONPATH
ENTRYPOINT ["python"]
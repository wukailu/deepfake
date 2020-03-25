FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN rm /etc/apt/sources.list.d/nvidia-ml.list && rm /etc/apt/sources.list.d/cuda.list
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && apt-get clean \
  && apt-get --allow-unauthenticated update \
  && apt-get install -y wget git libsm6 libxext6 libxrender-dev libgtk2.0-dev \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get --allow-unauthenticated update \
  && apt-get upgrade -y \
  && apt-get install bzip2

COPY miniconda.sh miniconda.sh
RUN bash miniconda.sh -b -p /miniconda
RUN /bin/bash -c "source ~/.bashrc" && conda init && /bin/bash -c "source ~/.bashrc"
RUN conda config --set ssl_verify no && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/ && \
 conda config --set show_channel_urls yes && conda remove pyyaml -y &&\
 conda install python=3.7.1

COPY apex/ apex/
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--pyprof" apex/

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
        && rm /tmp/requirements.txt

RUN export PYTHONPATH=/job/job_source/:$PYTHONPATH
ENTRYPOINT ["python"]
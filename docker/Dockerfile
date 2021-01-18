FROM nvidia/cuda:11.0-base

COPY content /app

RUN apt-get update && apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 gcc g++ && rm -rf /var/lib/apt/lists/*

# RUN mkdir /app
WORKDIR /app
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
ENV HOME=/home/user
RUN chmod 777 /home/user
ENV CONDA_AUTO_UPDATE_CONDA=false 
ENV PATH=/home/user/miniconda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin 
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh && chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p ~/miniconda && rm ~/miniconda.sh && conda install -y python==3.8.1 && conda clean -ya
RUN conda install -y pytorch torchvision cudatoolkit=11.0 -c pytorch && conda clean -ya

# ENV CUDA_HOME=/usr/local/cuda-11.0

RUN cd /app/ctcdecode && pip install --no-cache-dir . && pip install --no-cache-dir editdistance

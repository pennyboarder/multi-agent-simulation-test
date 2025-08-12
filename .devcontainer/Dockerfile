FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends tzdata
# ENV TZ=Asia/Tokyo

# 環境変数を設定
ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
# 必要そうなライブラリをインストール

RUN apt-get update && apt-get install -y wget git
RUN apt-get update && apt install -y build-essential libbz2-dev libdb-dev \
  libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
  libncursesw5-dev libsqlite3-dev libssl-dev \
  zlib1g-dev uuid-dev tk-dev

# RUN mkdir tmp
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz
RUN tar xJf Python-3.11.9.tar.xz
WORKDIR /tmp/Python-3.11.9
RUN ./configure
RUN make
RUN make install
WORKDIR /

# install pytorch
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# install torch-geometric
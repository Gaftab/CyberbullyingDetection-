# hash:sha256:07fe597c8ee21e2047b7c3e05816f6b6924ea6c7df4cf6342d963d5c306799e6
FROM registry.codeocean.com/codeocean/miniconda3:4.6.14-python3.7.3-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy==1.18.1 \
        pandas==1.0.1 \
        scikit-learn==0.22.1 \
        scipy==1.4.1 \
    && conda clean -ya

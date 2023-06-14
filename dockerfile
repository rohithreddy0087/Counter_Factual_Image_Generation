FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN apt-get update
RUN apt-get install -y wget
RUN apt install g++
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Create a new Conda environment
RUN conda update -n base -c defaults conda
RUN conda create -y --name pytorch python=3.8

# Activate the Conda environment
RUN echo "source activate pytorch" >> ~/.bashrc
ENV PATH /root/miniconda/envs/pytorch/bin:$PATH
RUN /bin/bash -c "source activate pytorch"

# Install PyTorch and Jupyter Notebook
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
RUN conda install -c conda-forge matplotlib
RUN conda install click
RUN conda install ninja
RUN conda install pytorch-ignite
RUN conda install streamlit
RUN conda install -c conda-forge cudatoolkit-dev
RUN conda install -y jupyter

# Set up Jupyter Notebook configuration
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py

COPY /Contrastive-Counterfactuals-with-GAN /home/Contrastive-Counterfactuals-with-GAN

EXPOSE 8888

CMD ["/bin/bash", "-c", "jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root"]


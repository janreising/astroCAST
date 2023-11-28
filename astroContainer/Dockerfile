FROM ubuntu:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install necessary packages
RUN apt-get update && pt-get install -y python3.9 python3-pip wget make gcc git nano vim ffmpeg libsm6 libxext6 libgl1-mesa-glx

# Get astroCAST github and make /data directory
RUN mkdir /home/data
RUN cd /home && git clone https://github.com/janreising/astroCAST.git

# Install astroCAST and jupyerlab
RUN pip3 install astrocast[testing,documentation]

# Run command
CMD [ "/bin/bash" ]

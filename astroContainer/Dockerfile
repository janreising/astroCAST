FROM almalinux:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install necessary packages
RUN dnf update -y && dnf install -y python3 python3-pip wget make gcc git nano vim ffmpeg libXext libSM mesa-libGL

# Get astroCAST github and make /data directory
RUN mkdir /home/data
RUN cd /home && git clone https://github.com/janreising/astroCAST.git

# Install astroCAST and jupyterlab
RUN pip3 install astrocast[testing,documentation]

# Run command
CMD [ "/bin/bash" ]

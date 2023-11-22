FROM ubuntu:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install necessary packages
RUN apt-get update
RUN apt-get install -y python3.9 python3-pip
RUN apt-get install -y wget make gcc git nano vim

# Get astroCAST github and make /data directory
RUN mkdir /home/data
RUN cd /home && git clone https://github.com/janreising/astroCAST.git

# Install astroCAST and jupyerlab
RUN pip3 install astrocast[all]

# Run command
CMD [ "/bin/bash" ]

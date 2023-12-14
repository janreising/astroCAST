FROM almalinux:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install necessary packages
RUN dnf update -y && dnf install -y python3 python3-pip python3-devel wget make gcc git nano vim libXext libSM mesa-libGL

# Install ffmpeg
#RUN dnf -y install https://download.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
#RUN yum-config-manager --enable powertools
#RUN dnf install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm -y
#RUN dnf install --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm -y
#RUN dnf install -y ffmpeg

# Get astroCAST github and make /data directory
RUN mkdir /home/data
RUN cd /home && git clone https://github.com/janreising/astroCAST.git

# Install poetry and install astrocast
RUN pip3 install poetry
RUN cd /home/astroCAST && poetry install -E testing

RUN echo "alias astrocast='poetry run astrocast'" >> ~/.bashrc \
	&& echo "alias python='poetry run python'" >> ~/.bashrc \
	&& echo "alias jupyter-lab='poetry run jupyter-lab'" >> ~/.bashrc \
	&& echo "alias pytest='poetry run pytest'" >> ~/.bashrc

# Install astroCAST and jupyterlab
#RUN pip3 install astrocast[testing,documentation]

# Change working directory
WORKDIR /home/astroCAST

# Run command
CMD [ "/bin/bash" ]

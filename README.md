# astroCAST

astroCAST is a Python package designed for analyzing calcium fluorescence events in astrocytes. It provides various functionalities to process and explore calcium imaging data, identify events, and perform spatiotemporal analysis.

## Features

- Fast data preparation
- Efficient event detection algorithm
- Visualization tools for exploring calcium imaging data and events
- Calculation of spatial statistics and cluster analysis
- Modularity detection and analysis of astrocyte modules

## Summary
Astrocytic calcium event analysis is challenging due to its complex nature, spatiotemporally overlapping events, and variable lengths. Our Astrocytic Calcium Signaling Toolkit (astroCAST) addresses these challenges by efficiently implementing event detection and variable length clustering. Leveraging dynamic thresholding, astroCAST captures diverse calcium fluctuations effectively. Designed for modularity, parallelization, and memory-efficiency, it enables fast and scalable event detection on large datasets.

## Table of contents
1. [Installation](#installation)
2. [Documentation](#documentation)
3. [Containers](#containers)
    1. [Docker container](#docker-container)
    2. [Singularity container](#singularity-container)
4. [Contributing](#contributing)

## Installation

**Conda:**
We recommend to first create a fresh conda environment to prevent conflicts with other packages.
```shell
conda create -n astrocast python=3.9
conda activate astrocast
```

**Via pip:**
```shell
pip install astrocast
```

**From source:**
```shell
git clone git@github.com:janreising/astroCAST.git
cd astroCAST
pip install poetry
poetry install
```

**With optional features:**
```shell
pip install astrocast[all]
# or using poetry
poetry install -E all
```

## Containers<a name="containers">
Now astroCAST implementation can run as stand-alone docker and singularity images.

Currently, we only support astroCAST in MacOS through docker, but other operating systems compatible docker may as well run astroCAST docker container. For more infomation on how to install docker and create an account, please visit https://docs.docker.com/engine/install/. For more information on how to install singularity(apptainer) and create an account visit https://apptainer.org/docs/admin/main/installation.html.

### Docker container<a name="docker-container">
Once docker has been installed, run the following commands in the terminal:
```shell
docker pull anacgon/astrocast:latest
```
This may take some minutes as the image is directly pulled from dockerhub. Once the image has been pulled locally, make sure it was correctly fetched by running:
```shell
docker image ls
```
You should be able to see the docker image listed.

To start a container using the image run:
```shell
docker run -v /path/to/your/data:/home/data -it -p 8888:8888 anacgon/astrocast:latest 
```
Note: "/path/to/your/data" must be replaced with your local path to the data you will use for the analysis. -p option allows the container to expose port 8888, necessary to run access jupyterlab from your browser.

To start a new jupyterlab notebook from inside the docker container, run:
```shell
jupyter-lab --allow-root --no-browser --port=8888 --ip="*"
```
Note: the explorer window will not be open automatically, you must copy and paste the URL provided in the console.

### Singularity container<a name="singularity">
Singularity (apptainer) now supports direct conversion of docker images into singularity. To pull and run the image use the following commands:
Pull singularity image directly from dockerhub.
```shell
singularity pull docker://anacgon/astrocast:latest
```
Once image has been pulled, a SIF file will be created in the directory where the previous command was executed. To run the singularity image:

```shell
singularity run --writable-tmpfs astrocast_latest.sif
```
note: singularity automatically mounts the host file system, therefore manual mounting is not required.

To start a new jupyterlab notebook from inside the singularity container, run:
```shell
jupyter-lab --ip "*" --no-browser
```
Please note that for this option to work, you will need to forward port 8888 (or customized port) of the cluster to your local computer when connecting via ssh. For example:

```shell
ssh -L 8888:host-node:8888 username@host-node
```

This will allow port 8888 in the host node to be forwarded to port 8888 in your local computer. 
Note: the explorer window will not be open automatically, you must copy and paste the URL provided in the console.

For more detailed examples and usage instructions, please refer to to the companion paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4491483) (currently in preprint).

## Documentation<a name="documentation">
Our [documentation](https://astrocast.readthedocs.io) is hosted on readthedocs.

## Contributing<a name="contributing">
Contributions to astroCAST are welcome! If you encounter any issues, have suggestions, or would like to contribute new features, please submit a pull request or open an issue in the GitHub repository.

## License
astroCAST is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact
For any inquiries or questions, please contact [Jan Philipp Reising](mailto:jan.reising@ki.se) or [Ana Cristina González](mailto:ana.cristina.gonzalez.sanchez@ki.se).

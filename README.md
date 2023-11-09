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
3. [Docker container](#docker-container)
4. [Contributing](#contributing)

## Installation<a name="installation">
You can install astroCAST using pip: 
```shell 
pip install astrocast[all]
```

Alternatively, clone this repository and install it locally:
```shell
pip install poetry
poetry install --with vis
```

## Docker container<a name="docker-container">
Currently, we support astroCAST in MacOS through docker, but other operating systems supporting docker may as well run astroCAST docker container. For more infomation on how to install docker and create an acconsult, please refer to https://docs.docker.com/engine/install/.

Once docker has been installed, run the following commands in the terminal:
```shell
docker pull anacgon/astrocast:1.1
```
This may take some minutes as the image is pulled from docker hub. Once the image has been pulled locally, make sure it was correctly fetched by running:
```shell
docker image ls
```
You should be able to see the docker image listed.

To run the image use:
```shell
docker run -v /path/to/your/data:/home/data -it -p 8888:8888 astrocast:1.1 
```
Note: "/path/to/your/data" must be replaced with your local path to the data you will use for the analysis. -p option allows the container to expose port 8888, necessary to run jupyterlab inside the container.

To start a new jupyterlab session from inside the container, run:
```shell
jupyter-lab --allow-root --no-browser --port=8888 --ip="*"
```
Note: the explorer window will not be open automatically, instead you must copy and paste the URL provided in the console.

For more detailed examples and usage instructions, please refer to to the companion paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4491483) (currently in preprint).

## Documentation<a name="documentation">

[//]: # (The documentation for astroCAST can be found here.)
We are currently working on the documentation. Please consult the `notebooks/` folder for example usage in the meantime.

## Contributing<a name="contributing">

Contributions to astroCAST are welcome! If you encounter any issues, have suggestions, or would like to contribute new features, please submit a pull request or open an issue in the GitHub repository.

## License

astroCAST is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact

For any inquiries or questions, please contact [Jan Philipp Reising](mailto:jan.reising@ki.se) or [Ana Cristina González](mailto:ana.cristina.gonzalez.sanchez@ki.se).

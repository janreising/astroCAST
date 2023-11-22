Installation
============

We recommend to first create a fresh conda environment to prevent conflicts with other packages.

.. code-block:: shell

    conda create -n astrocast python=3.9
    conda activate astrocast

.. attention::
    MacOS users with M1 chips may encounter compatibility issues during installation.
    For an effective workaround, please see the `Installation with containers`_ section.

Installation via pip
---------------------
.. code-block:: shell

    pip install astrocast
    # with optional features
    pip install astrocast[all]

Installation from source
-------------------------
.. code-block:: shell

    # install necessary packages
    pip install poetry

    # clone astroCAST repository
    git clone git@github.com:janreising/astroCAST.git
    cd astroCAST

    # install
    poetry install
    ## with optional features
    poetry install -E all

.. _installation-with-containers:

Installation with containers
----------------------------

astroCAST is now available as stand-alone Docker and Singularity (Apptainer) images, streamlining deployment and usage.

.. note::

   Docker support for astroCAST is verified on macOS. Users on other operating systems with Docker compatibility may also be able to run the astroCAST Docker container. Refer to the `Docker installation guide <https://docs.docker.com/engine/install/>`_ for setup instructions. For Singularity (Apptainer), consult the `installation documentation <https://apptainer.org/docs/admin/main/installation.html>`_.

Docker Container
*****************

After installing Docker, execute the following command to pull the astroCAST image (download may take a few minutes). Then verify the image pull by listing the Docker images.

.. code-block:: shell

    docker pull anacgon/astrocast:latest
    docker image ls

The astroCAST image should appear in the list. To initiate a container from the image, use:

.. code-block:: shell

   docker run -v /path/to/your/data:/home/data -it -p 8888:8888 astrocast:1.1

.. note::

   Replace "/path/to/your/data" with the actual data directory. The `-p` flag maps port 8888 to access JupyterLab via a web browser.

Within the Docker container, launch JupyterLab with:

.. code-block:: shell

   jupyter-lab --allow-root --no-browser --port=8888 --ip="*"

You'll need to manually navigate to the provided URL in your web browser to access JupyterLab.

Singularity Container
**********************

Singularity (Apptainer) can convert Docker images for use with Singularity. Pull the image with:

.. code-block:: shell

   singularity pull docker://anacgon/astrocast:latest

This command creates a SIF file in the current directory. Run the Singularity image with:

.. code-block:: shell

   singularity run --writable-tmpfs astrocast_latest.sif

Singularity automatically mounts the host filesystem; manual mounting is unnecessary.

To launch JupyterLab inside the Singularity container, enter:

.. code-block:: shell

   jupyter-lab --ip "*" --no-browser

For remote access, forward the port 8888 through SSH. This forwards port 8888 from the host node to your local machine.

.. code-block:: shell

   ssh -L 8888:host-node:8888 username@host-node

.. note::

   The JupyterLab interface won't open automatically. Copy and paste the console-provided URL into your browser.

For comprehensive examples and detailed instructions, consult the companion preprint paper `<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4491483>`_.

.. toctree::
   :maxdepth: 1

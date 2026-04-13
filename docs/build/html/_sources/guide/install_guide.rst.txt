Installation
============

This document provides step-by-step instructions for deploying and installing the StarryGL. Follow the steps below to ensure a successful installation.

Prerequisites
=============

Before starting the installation, make sure you meet the following prerequisites:

- Operating System:  Only Support for Linux.
- Python Version: Requires Python 3.10.
- CUDA Toolkit Version: Requires CUDA Toolkit 11.8.

Installation Steps
==================

1. Clone the StarryGL Git Repository
-------------------------------------

First, clone the StarryGL Git repository, including the submodule dependencies. Run the following command to clone the repository:

.. code-block:: bash

    $ git clone --recursive http://115.236.33.124:7001/wjie98/starrygl.git

If you have already cloned the repository without the submodules, you can run the following command to download the submodules:

.. code-block:: bash

    $ git submodule update --init --recursive

This will ensure that all the necessary submodule dependencies are downloaded.

2. Install Dependencies
-----------------------

Before installation, ensure that the necessary dependencies are installed on your system. Run the following command to install these dependencies:

.. code-block:: bash

    $ pip install -r requirements.txt

3. Execute the Installation Script
----------------------------------

Locate the installation script file `install.sh` in the installation directory and execute the following command to install StarryGL:

.. code-block:: bash

    $ bash ./install.sh

4. Install StarryGL
------------------------

The installation script will execute the `python setup.py install` command to install StarryGL. Make sure you are in the installation directory of StarryGL and execute the following command:

.. code-block:: bash

    $ python setup.py install

5. Verify the Installation
--------------------------

After the installation, you can verify if StarryGL is successfully installed by executing the following command:

.. code-block:: bash

    $ python -c "import starrygl; print(starrygl.__version__)"

If the StarryGL version is successfully displayed, it means the installation was successful.

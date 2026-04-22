Installation & Build Guide
==========================

This guide reflects the current repository layout and the build flow verified in the
current development environment.

StarryUniGraph uses ``pyproject.toml`` for Python packaging and the repository-root
``CMakeLists.txt`` to build the native C++ extensions.

Clone the Repository
--------------------

If you already have a local checkout, enter the repository root and continue.
Otherwise clone the repository from your current remote:

.. code-block:: bash

    git clone https://github.com/zhljJoan/starrygl-unidynGraph.git
    cd starrygl-unidynGraph

Verified Environment
--------------------

The commands in this document were checked in the following environment:

- Linux
- Python 3.10.13
- PyTorch 2.1.1+cu118
- DGL 1.1.3+cu118
- CMake 3.16.3
- GNU C++ 9.4.0

Prerequisites
-------------

Before building, make sure that:

- you use **Python 3.10 or newer**
- PyTorch and DGL are already installed in the same environment
- PyTorch, DGL, and CUDA are version-compatible
- ``cmake`` and a C++17-capable compiler are available
- you build the native extensions with the **same Python interpreter** that will run the package

.. The current repository does **not** provide a ``requirements.txt`` or ``setup.py`` based
.. install flow. Do not use ``python setup.py install`` here.

Check the Active Environment
----------------------------

Run these checks before building:

.. code-block:: bash

    python --version
    which python
    cmake --version
    c++ --version

    python - <<'PY'
    import sys
    import torch
    import dgl
    import yaml
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("dgl:", dgl.__version__)
    print("yaml:", yaml.__version__)
    PY

Build and Install
-----------------

1. Activate your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example:

.. code-block:: bash

    conda activate starrygl_graph

2. Build the native C++ extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always pass the active interpreter explicitly:

.. code-block:: bash

    cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
    cmake --build build -j

This step builds the native modules into ``starry_unigraph/lib/``:

- ``starry_unigraph/lib/libstarrygl_sampler.so``
- ``starry_unigraph/lib/adaptive_split_cpp.so``

3. Install the Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m pip install -e .

4. Verify the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the CLI entry point:

.. code-block:: bash

    python -m starry_unigraph -h

Check the native modules:

.. code-block:: bash

    python - <<'PY'
    import importlib
    from starry_unigraph.native import is_bts_sampler_available

    importlib.import_module("starry_unigraph.lib.adaptive_split_cpp")
    print("bts_sampler_available=", is_bts_sampler_available())
    print("adaptive_split_cpp=OK")
    PY

If the native build is correct, ``is_bts_sampler_available()`` should print ``True``.

Native Build Entry Points
-------------------------

The main C++ build entry is the repository root:

- ``CMakeLists.txt``

The two native module source entry points are:

- ``starry_unigraph/vendor/bts_sampler/sampler/export.cpp``
- ``starry_unigraph/vendor/bts_sampler/split/time_split.cpp``

At runtime, these modules are loaded from:

- ``starry_unigraph/lib/loader.py``
- ``starry_unigraph/native/bts_sampler.py``
- ``starry_unigraph/backends/chunk/prepare/time_split.py``

Quick Start
-----------

Before running the example configs, update the paths inside the YAML files first.
Several configs in ``configs/`` currently use machine-specific absolute paths such as
``/mnt/...`` for datasets and checkpoints.

For example: preprocess first, then train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m starry_unigraph --config configs/mpnn_lstm_4gpu.yaml --phase prepare
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase train

Notes
-----

- ``python -m starry_unigraph`` is the current CLI entry point.
- Use ``torchrun`` for multi-GPU runs.
- For distributed workflows, run ``prepare`` and ``train`` as separate steps.
- The package metadata lives in ``pyproject.toml``.

.. Troubleshooting
.. ---------------

.. Python rejected by pip
.. ~~~~~~~~~~~~~~~~~~~~~~

.. If you see:

.. .. code-block:: text

..     ERROR: Package 'starry-unigraph' requires a different Python: 3.9.x not in '>=3.10'

.. switch to Python 3.10+ and reinstall.

.. Native module built with the wrong Python
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. If you see:

.. .. code-block:: text

..     ImportError: Python version mismatch: module was compiled for Python 3.9,
..     but the interpreter version is incompatible: 3.10...

.. rebuild the native modules with the interpreter you are actually using:

.. .. code-block:: bash

..     cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
..     cmake --build build -j

.. CMake uses the wrong interpreter
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. The top-level ``CMakeLists.txt`` contains a fallback Python path. To avoid compiling
.. against the wrong environment, always pass:

.. .. code-block:: bash

..     -DPython3_EXECUTABLE=$(which python)

.. BTS sampler is unavailable
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~

.. If ``is_bts_sampler_available()`` returns ``False``:

.. - confirm that ``starry_unigraph/lib/libstarrygl_sampler.so`` exists
.. - rebuild the native extensions
.. - make sure PyTorch and the native modules were built in the same environment

See Also
--------

- :doc:`../rel_0_1_1/intro` — Project background and graph-mode overview
- :doc:`../rel_0_1_1/training/index` — Training guide index
- :doc:`../rel_0_1_1/training/dtdg` — DTDG training workflow
- :doc:`../rel_0_1_1/training/ctdg` — CTDG training workflow
- :doc:`../rel_0_1_1/training/chunk` — Experimental chunk workflow
- :doc:`../architecture/__init__` — Architecture reference

.. _dev-install:

========================
Development Installation
========================

- Add user to HiAER-Spike system and github repos, and CRISDSC user group on servers

- If using the HiAER-Spike system ensure default shell is set to bash
  .. code-block:: bash

     chsh -s /bin/bash

Install Pyenv
-------------


- Install `Pyenv <https://github.com/pyenv/pyenv>`_

  .. code-block:: bash

     curl https://pyenv.run | bash

- Install python 3.10

  .. code-block:: bash

    pyenv install 3.10

- Set python 3.10 as the global version before installing poetry

  .. code-block:: bash
                  
    pyenv global 3.10

Install Poetry
--------------

- Install `Poetry <https://python-poetry.org/>`_

  .. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

- Set Poetry to use local venv

  .. code-block:: bash

    poetry config virtualenvs.in-project true

- Add shell extension for Poetry

  .. code-block:: bash

    poetry self add poetry-plugin-shell

Install Libraries
-----------------

- Install `Fxpmath fork <https://github.com/Integrated-Systems-Neuroengineering/fxpmath>`_

  .. code-block:: bash

    git clone git@github.com:Integrated-Systems-Neuroengineering/fxpmath.git
    cd fxpmath
    pyenv local 3.10
    poetry install

- Install  `connectome_utils <https://github.com/Integrated-Systems-Neuroengineering/connectome_utils>`_

  .. code-block:: bash

    git clone https://github.com/Integrated-Systems-Neuroengineering/connectome_utils
    cd connectome_utils
    git checkout dev
    pyenv local 3.10
    poetry install


- Install  `hs_bridge <https://github.com/Integrated-Systems-Neuroengineering/hs_bridge>`_

  .. code-block:: bash

    git clone git@github.com:Integrated-Systems-Neuroengineering/hs_bridge.git
    cd hs_bridge
    git checkout dev
    pyenv local 3.10
    poetry install

- Compile the cython components of hs_bridge

  .. code-block:: bash

    poetry shell
    cd hs_bridge/wrapped_dma_dump
    make

- Install `hs_api <https://github.com/Integrated-Systems-Neuroengineering/hs_api>`_

  .. code-block:: bash

    git clone git@github.com:Integrated-Systems-Neuroengineering/hs_api.git
    cd hs_api
    git checkout dev
    pyenv local 3.10
    poetry install

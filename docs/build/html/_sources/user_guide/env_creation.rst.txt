Environment Creation and Installation
=====================================

#. Create a new environment following the steps:
    
    * Open the mambaforge terminal ('Miniforge Prompt') and type:

        .. code-block:: console

            (base) $ mamba create -n circadipy_env python=3.8

        .. note::

           Python version must be 3.8, because PyMICE, one of the dependencies of circadipy to read Intellicage
           data, is not compatible with versions of Python higher than 3.8.

    * Press 'y' when prompted.

    * After the environment is created, type:

        .. code-block:: console

            (base) $ mamba activate circadipy_env

#. Install the dependences from PyPI repository by typing: 

    .. code-block:: console 

        (circadipy_env) $ pip install circadipy

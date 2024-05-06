Working With Jupyter Notebooks
==============================

Jupyter notebooks are a great way to work with Python code by allowing code editing and running directly in a web browser.
Additionaly, it offers the inclusion of formatted text, images and interactive visualizations inside of the notebook.
Thus it is a great tool for data analysis.

To create a new notebook, follow these steps:

#. Open the 'Miniforge Prompt' from the Start menu.

#. Activate the environment created in :doc:`Environment Creation section<env_creation>` by typing:

   .. code-block:: console

      (base) $ mamba activate circadipy_env

#. Navigate to the directory where you want to create the notebook. For example, to create a notebook in the 'my_name/Documents' 
   folder, you would type:

   .. code-block:: console

        (circadipy_env) $ cd my_name/Documents

#. Start the Jupyter notebook server by typing:

   .. code-block:: console

        (circadipy_env) $ jupyter notebook

#. This will open a new tab in your web browser with the Jupyter notebook interface. Click on the 'New' button at the top
   right corner and select 'Python 3' to create a new notebook. At this point, you can start writing and running Python code
   using the CircadiPy library.

----------------------------------------------------------------------------------------------------------------------------

You can also run the tutorial notebook provided in the CircadiPy package (../circadipy/src/circadipy/tutorial_pipeline.ipynb). 
To do this, follow these steps:


Using the tutorial notebook
---------------------------

#. Open the 'Miniforge Prompt' from the Start menu.

#. Activate the environment created in :doc:`Environment Creation section<env_creation>` by typing:

   .. code-block:: console

      (base) $ mamba activate circadipy_env

#. Run Python and import the CircadiPy and pathlib modules. Then, use the pathlib module to find the path to the tutorial
   notebook typing the following commands:

   .. code-block:: python 

      (circadipy_env) $ python
      >>> import circadipy
      >>> from pathlib import Path
      >>> Path(circadipy.__file__).parent / 'tutorial_pipeline.ipynb'

#. Get the path to the tutorial notebook and copy it without the quotes. You will use this path in the next step

#. Exit the Python interpreter by typing:

   .. code-block:: python

      >>> exit()

#. Start the Jupyter notebook server by typing "jupyter notebook" + the copied path:

   .. code-block:: console

        (circadipy_env) $ jupyter notebook copied_path_here
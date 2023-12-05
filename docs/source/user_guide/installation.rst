Installation
============

The installation process for the package is pretty straightforward with one catch. You need to first install Microsoft Visual C++ Build Tools.
This is to ensure that, in the installation process, the PyMice package is installed correctly. PyMice is a dependency of CircadiPy.


Installing Microsoft Visual C++ Build Tools
-------------------------------------------

#. Go to the `Microsoft Visual C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ website.
#. There, download the installer for the Build Tools for Visual Studio as show in this picture:

   .. image:: ../imgs/Microsoft_build_tools_site.png
        :scale: 50 %
        :align: center

#. On the installer you need to select C++ Build tools on the left	(Desenvolvimento para desktop com C++ em portugues)
#. On the right, select only "MSVC v143 build tools" and "Windows 11 SDK"

    .. image:: ../imgs/Microsoft_build_tools_selection.png
        :scale: 50 %
        :align: center

#. After the installation you can continue with circadipy`s installation process normally using the requirements.txt


After this step is completed, you can follow these steps:
---------------------------------------------------------

#. Make sure that the environment you created in the previous section is activated (the terminal should look like this):

   .. code-block:: console

       (circadipy_env) $

   * If it is not activated, you can activate it using the following command:

   .. code-block:: console

       (base) $ mamba activate circadipy_env

#. Then you can install the package using pip:

   .. code-block:: console

       (circadipy_env) $ pip install circadipy

#. Wait for the installation to finish.
#. After the installation you can install the requirements using the requirements.txt provided.

   .. code-block:: console

       (circadipy_env) $ pip install -r requirements.txt
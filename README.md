# Welcome to Circadipy!
=======================================

Introducing **CircadiPy**, a Python package for chronobiological analysis! 
With seamless integration of powerful time series plotting libraries, 
it enables researchers to visualise and study circadian cycles with unrivalled versatility.

Currently, the package supports the visualisation of biological rhythms and their synchronisation with external cues using

1. Actograms: An actogram is a graphical representation of an organism's activity or physiological data over time. It typically shows activity or physiological measurements (e.g. hormone levels, temperature) on the y-axis and time on the x-axis. Actograms are often used to visualise circadian rhythms and patterns of activity/rest cycles.

2. Cosinor Analysis Plot: This plot is used to analyse and display the presence of rhythmic patterns in the data. It's a graphical representation of cosinor analysis, which fits a cosine curve to the data to estimate rhythm parameters such as amplitude, acrophase (peak time) and period.

3. Raster plot: A raster plot shows individual events or occurrences (such as action potentials in neurons) over time. In chronobiology this can be used to show the timing of specific events in relation to the circadian cycle.

4. Histogram: A histogram can be used to show the distribution of events or measurements over a period of time. In chronobiology this could be the distribution of activity bouts or physiological measurements over different time bins.

------------------------------------------------------------------------------------------------------------------------------

# CircadiPy installation:
    * This guide is in a simplified format, for a more detailed guide and much more please visit our documentation page: https://circadipy.readthedocs.io/en/latest/


## Before Installation

### Install Visual Studio Build Tools

CircadiPy requires the installation of the PyMICE library to read Intellicage data. Additionally, PyMICE requires the installation of Visual Studio Build Tools on Windows. You can download the Visual Studio Build Tools from the following link:

[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

After downloading the Visual Studio Build Tools, you can install it by running the installer. Once the installation is completed, you can proceed to the following steps.

1. Select the "desktop development with C++" workload.

    ![Build Tools](https://raw.githubusercontent.com/nnc-ufmg/circadipy/main/docs/source/imgs/build_tools.png)

2. On the right side of the window, select only the "MSVC" (e.g. "MSVC v142 - VS 2019 C++ x64/x86 build tools") and "Windows X SDK" components.

3. Click on the "Install" button to start the installation.

------------------------------------------------------------------------------------------------------------------------------

# Package and Environment Management System

We recommend using either Miniconda or Mamba to install Python and create a new environment. In this tutorial, we will use Mamba, but the steps are the same for Miniconda.

To install Python using Mamba, you need the Mambaforge installer. You can download it from the [Mambaforge GitHub page](https://github.com/conda-forge/miniforge#mambaforge).

There, scroll to the Mambaforge section and download the installer for your operating system. This will install the Mambaforge package and environment management system with Mamba installed in the base environment, as shown in this image:

<p align="center">
    <img src="https://raw.githubusercontent.com/nnc-ufmg/circadipy/main/docs/source/imgs/mambaforge_github_page.png" alt="Mambaforge GitHub Page" />
</p>

*Figure: The section where the user will download the Mambaforge installer. You can observe that there are options for different operating systems, including Linux, macOS, and Windows. The user should download the installer for their operating system. In this case, download the Windows installer, `Mambaforge-Windows-x86_64`.*

After the download is completed, run the installer and follow the instructions. Once the installation is completed, a new program called "Miniforge Prompt" will appear in your start menu. (Alternatively, you can type "miniforge" into your operating system's search bar to find the prompt.) Open it and use the Mambaforge terminal to create a new environment with the required dependencies.

------------------------------------------------------------------------------------------------------------------------------

# Environment Creation and Installation

1. Create a new environment by following these steps:

    - Open the Mambaforge terminal (also known as 'Miniforge Prompt') and type:

        ```console
        (base) $ mamba create -n circadipy_env python=3.8
        ```

        > **Note**: Python version must be 3.8 because PyMICE, a dependency of CircadiPy for reading Intellicage data, is not compatible with Python versions higher than 3.8.

    - Press 'y' when prompted.

    - After the environment is created, activate it by typing:

        ```console
        (base) $ mamba activate circadipy_env
        ```

2. Install the dependencies from the PyPI repository by typing:

    ```console
    (circadipy_env) $ pip install circadipy
    ```

------------------------------------------------------------------------------------------------------------------------------

### CircadiPy also has a built-in simulated data generator that allows the creation of custom data sets for testing, experimentation and comparison purposes.

### Please visit our documentation page for more information: https://circadipy.readthedocs.io/en/latest/ 

There you will find:

1. A more in depth guide on how to use CircadiPy using the simulated data generator in the tutorial section
2. How to use the package to analyse your own data leveraging Jupiter notebooks.
3. A description of the API 


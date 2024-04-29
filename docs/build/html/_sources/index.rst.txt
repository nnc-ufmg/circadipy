.. circadipy documentation master file, created by
   sphinx-quickstart on Tue Aug  1 12:37:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to circadipy's documentation!
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

CircadiPy also has a built-in simulated data generator that allows the creation of custom data sets for testing, experimentation and comparison purposes.

.. note::

   You can view the whole source code for the project on
   `Circadipy's Github page <https://github.com/nnc-ufmg/circadipy>`_


.. toctree::
   :caption: USER GUIDE
   :maxdepth: 2
   :hidden:

   user_guide/before_install
   user_guide/package_manager
   user_guide/env_creation
   user_guide/working_with_ipynb
   user_guide/pipelines

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   api_reference/api_description
   api_reference/chrono_plotter
   api_reference/chrono_reader
   api_reference/chrono_simulation
   api_reference/chrono_rhythm
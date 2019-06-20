============
submovements
============


.. image:: https://img.shields.io/pypi/v/submovements.svg
        :target: https://pypi.python.org/pypi/submovements

.. image:: https://img.shields.io/travis/sstbrg/submovements.svg
        :target: https://travis-ci.org/sstbrg/submovements

.. image:: https://readthedocs.org/projects/submovements/badge/?version=latest
        :target: https://submovements.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/sstbrg/submovements/shield.svg
     :target: https://pyup.io/repos/github/sstbrg/submovements/
     :alt: Updates



Our goal is to detect sub-movements from motion data.


* Free software: MIT license
* Documentation: https://submovements.readthedocs.io.


Introduction
---------------

.. image:: https://ars.els-cdn.com/content/image/1-s2.0-S2352914818302028-gr5.jpg
    :target: https://ars.els-cdn.com/content/image/1-s2.0-S2352914818302028-gr5.jpg
    :caption: https://doi.org/10.1016/j.imu.2019.01.005

This software works on a pipeline basis.

To install use: pip install submovements

Input
~~~~~~~~~~~~~~~
The input is a directory of trials which are saved as CSV files with the following file names:
li_stimulus_side_block#_repetition#.csv

Other .csv files with a different name format **are ignored**.

Pre-processing (Preprocessor class)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. (x,y) coordinates as a function of time are extracted from every CSV file.

2. Butterworth zero-phase of 4th order is applied on the (x,y) positions. Further filtering is possible by expanding the preprocessor class.

3. d(x,y)/dt is calculated to yield velocities (Vx,Vy) as a function of time.

4. To remove the duration where (Vx,Vy) are approximately zero we use thresholding on ||(Vx,Vy)||, such that the any data where ||(Vx,Vy)|| < threshold (0.001 by default) is removed, **not including** a 0.1s portion around the time where motion took place.

5. The filtered velocity is saved under the Trial class for further processing.

*For specific explanations on methods and attributes see commentary in submovements/DataProcessing.py.*

Trial processing (Trial class)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class represents a single Trial for a given subject (labeled by its numeric id).
For example, if the CSV files are saved under:
:: data/results/sinmonvisual/12345/li_stimulus_side_block#_repetition#.csv
The subject id is 12345

Additional attributes are: stimulus, block, repetition and data.

*For specific explanations on methods and attributes see commentary in submovements/DataProcessing.py.*

Credits
-------


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

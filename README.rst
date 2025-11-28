drillcore-transformations
=========================

A minimal and simple Python library for transforming drillcore alpha,
beta, and gamma measurements.

|PyPI Status| |CI Test|

Usage
-----

Transform drillcore alpha, beta, and gamma measurements to planar and
linear orientations:

.. code:: python

   from drillcore_transformations import transform

   # Example measurement values
   alpha = 14
   beta = 42
   drillcore_trend = 213
   drillcore_plunge = -85
   gamma = 53  # Optional

   # Transform the measurements
   dip, direction, plunge, trend = transform(
       alpha,
       beta,
       drillcore_trend,
       drillcore_plunge,
       gamma
   )

   print(f"Dip: {dip}, Direction: {direction}, Plunge: {plunge}, Trend: {trend}")
   # Results:
   # Dip: 82.0, Direction: 75.0, Plunge: -22.0, Trend: 176.0

Convention
----------

- Negative ``drillcore_plunge`` means the drillcore is pointing downwards.
- Negative ``plunge`` result also means the linear feature is pointing
  downwards.
- ``beta`` is measured from the bottom mark clockwise, while looking toward
  the end of the hole.
- ``gamma`` is assumed to be vector data (direction matters; 0–360), not
  axial (0–180).
- The ``gamma`` angle describes the orientation of a line within a plane,
  measured as the angle between the plane's long axis and the line
  itself. This library uses the 0–360° clockwise convention.

Development
-----------

To run tests:

.. code:: bash

   uv run pytest

Credits
-------

- PhD Jussi Mattila for tips, code snippets and sample materials.
- Authors of `Orientation uncertainty goes
  bananas <https://tinyurl.com/tqr84ww>`__ for great article and
  complementary excel-file.
- For conventions and procedures, see: Holcombe, Rod. *Oriented
  drillcore: measurement, conversion, and QA/QC procedures for
  structural and exploration geologists*. 2023.
  https://www.holcombe.net.au/downloads/HCOVG_oriented_core_procedures.pdf

License
-------

Copyright © 2025, Nikolas Ovaskainen.

--------------

.. |PyPI Status| image:: https://img.shields.io/pypi/v/drillcore-transformations.svg
   :target: https://pypi.python.org/pypi/drillcore-transformations
.. |Tests| image:: https://github.com/nialov/drillcore-transformations/actions/workflows/main.yaml/badge.svg
   :target: https://github.com/nialov/drillcore-transformations/actions/workflows/main.yaml

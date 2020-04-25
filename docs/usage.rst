=====
Usage
=====

To use Drillcore Transformations in a project:

.. code-block:: python

	import drillcore_transformations as dtp


Config
------

Columns in your data file are unlikely to be recognized by default. Therefore a config file exists that
can be edited with the column identifiers for any unique data file column names.

Another option is to modify your datafiles columns to match the default identifiers already in the config.ini.

.. csv-table:: **Default identifiers in config.ini**
   :header: "Measurement type", "Default identifier"
   :widths: 20, 20

   "Alpha", "alpha"
   "Beta", "beta"
   "Gamma", "gamma"
   "Borehole Trend", "borehole_trend"
   "Borehole Plunge", "borehole_plunge"
   "Measurement Depth", "measurement_depth"
   "Depth in Depth-file", "depth"


Manual editing of the config file is probably fastest but is risky.
The file can be found within the package with filename "config.ini". Path to
the config file can be printed with:

.. code-block:: python

	from dtp.usage import find_config
	find_config()

Editing the config can also be done with Python methods. This might prevent errors.
To make sure your column is identified properly
you can add the column as an identifier in the config with:

.. code-block:: python

	from dtp.usage import add_column_name
	add_column_name(header, base_column, name)

**Header** is the type of column name. Options for **header** are either:

	*"MEASUREMENTS" "DEPTHS" or "BOREHOLE"*

**Base_column** is the type of data. Options for **base_column** are either:

	*"alpha" "beta" "gamma" "measurement_depth" "depth" "borehole_trend" or "borehole_plunge"*

**Name** is the column name that you wish to add to the identification list.

Example where you wish to add "alpha_measurements" as a new identifier
for an alpha measurement:

.. code-block:: python

	from dtp.usage import add_column_name
	add_column_name("MEASUREMENTS", "alpha", "alpha_measurements")

To reset the config file to defaults and **erase** all modifications:

.. code-block:: python

	from dtp.usage import initialize_config
	initialize_config()

Transforming data
------------------
Methods for transforming data in .csv or .xlsx format are found in
drillcore_transformations.usage:

.. code-block:: python

	from dtp.usage import \
	transform_csv, transform_excel, transform_csv_two_files, transform_excel_two_files

If data for measurements (e.g. alpha, beta, gamma) and data for depth variation
(e.g. depth, drillcore trend and plunge) are in separate files, use the
methods with _two_files suffix.
Otherwise use the methods without a suffix.

To transform a .csv file with column names that are or have been added in
the config file and with no gamma data:

.. code-block:: python

	transform_csv(filename="example.csv", output="example_transformed.csv", with_gamma=False)

This will save a new file "example_transformed.csv" to the same directory
your input file is in.

To transform .xlsx files where measurement data and depth+trend+plunge data
are separated and with gamma data:

.. code-block:: python

	transform_csv_two_files(measurement_filename="example.xlsx", depth_filename="example_depth_data.xlsx"
	, with_gamma=True, output="example_two_files_transformed.csv")

All "example_*" filenames should be replaced with your own filenames.

Basic transforming and visualization
-------------------------------------

Todo.


Measurement conventions
---------------------------------------------

Testing this module has been difficult due to high variance in the conventions
used to measure alpha and beta structures
from drillcores. To add to that, there's a high degree of variance in
the nomenclature of borehole/drillcore orientation
(e.g. for borehole trend = azimuth, bearing, trend; borehole plunge = dip, inclination; etc.)
Negative values are sometimes
only used to identify boreholes that have been drilled downhole i.e.
towards the center of the Earth.

**Currently this module uses this convention:**

	* Alpha is the angle between the discontinuity and the core axis.

	* Beta is measured clockwise from the reference line (orientation line) to the maximum dip vector and the reference line is at the bottom of the core.

	* Gamma measurements require further testing to define the convention with certainty....

	* Borehole/drillcore trend is the direction of plunge of the borehole/drillcore, between 0 and 360 degrees.

	* Borehole/drillcore plunge is the angle between the Earth's surface and the borehole/drillcore.

A crude method currently exists to test different conventions on data. This can be possibly used to find
the right conventions for your data though this is currently a "brute-forcing" method.

.. code-block:: python

	from dtp.usage import convention_testing_csv
	convention_testing_csv("your .csv file", with_gamma=True
	, visualize=False, output=r"your output file", img_dir=r"your directory for visualization images")

Visualization is only sensible if data only contains ~1 row. The convention cipher in output file columns is
as follows:

* - == negative
	* Meaning the sign of measurement is changed.

* 0 == none
	* Meaning no change is made to input measurement value.

* Order of measurement types in cipher:
	* [alpha, beta, borehole_trend, borehole_plunge, gamma]

* Example:
	* 0\|0\|0\|-\|-
	* Is equal to: no changes to alpha, beta, borehole_trend, but change of sign for borehole_plunge and gamma

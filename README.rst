Drillcore Transformations
=========================

|Documentation Status| |PyPI Status| |CI Test| |Coverage|

Features
--------

-  Transforms measurements from drillcores.
-  Supports alpha, beta and gamma measurements.
-  Supports .csv and .xlsx files.
-  Supports adding the column names of your data files to a custom-built
   config.ini file for each user.
-  TODO: Convention support
-  Currently supported convention explanation found in `Documentation
   and Help <https://drillcore-transformations.readthedocs.io>`__
-  **Documentation and Help**:
   https://drillcore-transformations.readthedocs.io.

Running tests
-------------

To run pytest in currently installed environment:

.. code:: bash

   poetry run pytest

To run full extensive test suite:

.. code:: bash

   poetry run invoke test

Formatting and linting
----------------------

Formatting and linting is done with a single command. First formats,
then lints.

.. code:: bash

   poetry run invoke format-and-lint

Building docs
-------------

Docs can be built locally to test that ``ReadTheDocs`` can also build
them:

.. code:: bash

   poetry run invoke docs

Invoke usage
------------

To list all available commands from ``tasks.py``:

.. code:: bash

   poetry run invoke --list

Development
~~~~~~~~~~~

Development dependencies include:

   -  invoke
   -  nox
   -  copier
   -  pytest
   -  coverage
   -  sphinx

Big thanks to all maintainers of the above packages!

Credits
-------

-  PhD Jussi Mattila for tips, code snippets and sample materials.
-  Authors of `Orientation uncertainty goes
   bananas <https://tinyurl.com/tqr84ww>`__ for great article and
   complementary excel-file.

License
~~~~~~~

Copyright © 2020, Nikolas Ovaskainen.

-----


.. |Documentation Status| image:: https://readthedocs.org/projects/drillcore-transformations/badge/?version=latest
   :target: https://drillcore-transformations.readthedocs.io/en/latest/?badge=latest
.. |PyPI Status| image:: https://img.shields.io/pypi/v/drillcore-transformations.svg
   :target: https://pypi.python.org/pypi/drillcore-transformations
.. |CI Test| image:: https://github.com/nialov/drillcore-transformations/workflows/test-and-publish/badge.svg
   :target: https://github.com/nialov/drillcore-transformations/actions/workflows/test-and-publish.yaml?query=branch%3Amaster
.. |Coverage| image:: https://raw.githubusercontent.com/nialov/drillcore-transformations/master/docs_src/imgs/coverage.svg
   :target: https://github.com/nialov/drillcore-transformations/blob/master/docs_src/imgs/coverage.svg

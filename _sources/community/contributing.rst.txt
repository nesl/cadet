Contributing
============

Contributions are welcome. The steps below cover branching, commit style, and
how to run the test suite before submitting.

See :doc:`../apps` for how to write a new app plugin, and
:doc:`../explanation/architecture` for an overview of the codebase structure.

----

Getting started
---------------

1. Clone the repository and create a branch from ``main``:

   .. code-block:: bash

      git checkout -b my-feature

2. Install the package in editable mode with development dependencies:

   .. code-block:: bash

      python -m venv .venv && source .venv/bin/activate
      pip install -e ".[test]"

3. Run the test suite before your first change to confirm a clean baseline:

   .. code-block:: bash

      pytest

----

Code conventions
----------------

- Follow the existing module docstring convention (see any file under
  ``cadet/``): what the file is for, what it contains, how it fits into
  CADET, how to configure it.
- No dates, personal names, hostnames, or IP addresses in code, comments,
  or commit messages.
- Public classes and functions need docstrings. Internal helpers do not.
- ``make -C docs html`` must pass with zero warnings (``-W`` is on).

----

Commit messages
---------------

Use short, neutral imperative sentences:

.. code-block:: text

   add netwaggle wired-link support
   fix netem teardown idempotency
   docs: expand network-backends page

The commit-msg hook in this repository rejects messages that contain dates
or attribution trailers.

----

Submitting
----------

Open a pull request against ``main``. The CI suite runs the tests and
builds the docs. A passing build is required before merge.

If your change adds a feature, record it in ``cadet/config/implemented.py``
with the run that verified it, and add its row to the matrix on the
:doc:`../index` home page.

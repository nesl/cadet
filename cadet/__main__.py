"""Module entry point so ``python -m cadet`` runs the CLI.

Delegates to :func:`cadet.cli.main`. See :mod:`cadet.cli` for the available
subcommands (currently ``validate``).
"""

import sys

from cadet.cli import main

if __name__ == "__main__":
    sys.exit(main())

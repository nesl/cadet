"""CADET role runtimes.

Generic client and server runtimes that drive an app plugin across the paper's
Fig. 1(b) stages. The app supplies the per-stage behaviour; these runtimes own
the loop, the source/transport wiring, and output-directory handling.

- :mod:`cadet.roles.client` — source -> process -> transport -> action -> metrics
- :mod:`cadet.roles.server` — transport -> frame queue -> active model -> reply
- :mod:`cadet.roles.load` — synthetic concurrent-client load generator

Per-frame CSV logging lives in :mod:`cadet.metrics`.
"""

from cadet.roles.client import ClientRuntime, RunContext, build_client
from cadet.roles.server import ServerRuntime, build_server

__all__ = ["ClientRuntime", "RunContext", "build_client", "ServerRuntime", "build_server"]

# The acceptance matrix

`matrix.yaml` is the specification: four layers (source x processing x network
x infrastructure), the cells that are valid, the ones that are n/a and why, and
the four checks every cell answers. `cells/` holds one config per cell and
`bases/` the configs they extend. `scripts/acceptance/run_matrix.py` runs them.

    python scripts/acceptance/run_matrix.py              # every valid cell
    python scripts/acceptance/run_matrix.py --cell carla.std   # a prefix
    python scripts/acceptance/run_matrix.py --layer network=delay

Outputs go under `--out` (a scratch directory by default), never into the tree.

## What the matrix needs from the machine it runs on

A cell's config describes the experiment. It cannot describe the machine, so
several values are deliberately placeholders, and a cell whose dependency is
absent is reported `blocked` rather than `fail`. Supply them like this:

| What | How | Placeholder in the spec |
|---|---|---|
| The simulator | `--set source.carla.host=<address>` | `host: CARLA_HOST` |
| The second machine | `$CADET_SECOND_MACHINE`, or `second_machine` in `local.yaml` | `host: edge-host` |
| Model weights | `--model-dir <dir>`, or `$CADET_MODEL_DIR` | - |
| The live stream | `--set source.url=<url>` | - |
| The analysis endpoint | `--set analysis.endpoint=<url>` and `$CADET_FM_API_KEY` | - |

**Set your second machine's address** before running the distributed cells:
either `export CADET_SECOND_MACHINE=<address>`, or copy
`local.yaml.example` to `local.yaml` and fill it in. `local.yaml` is untracked,
because an address belongs to a machine and not to this repository; it can also
carry the account, the key and the virtual environment where those differ from
the spec's defaults.

### A CARLA cell needs more than an address

The client role imports the simulator's own Python client, which is built for a
particular interpreter, so these cells run with an interpreter that can import
it - the one in `cadet:dev`, via `scripts/cadet`, rather than a host
interpreter of a different version. They also need the ScenarioRunner and CARLA
PythonAPI trees, named by `$CADET_SCENARIO_RUNNER_DIR` and
`$CADET_CARLA_PYTHONAPI_DIR`, and a scenario for the hook to drive. A scenario
is three pieces that have to agree: the file defining the scenario class
(`source.scenario.definition`), the file naming an instance of it
(`source.scenario.config_file`), and that instance's name
(`source.scenario.name`). With no scenario named, no sim hook is started, and a
CARLA cell then fails later with `no ego vehicle found`.

## What the checks mean

- **(a) completes and tears down** - the run reached a recorded end, the
  client's own outcome says it succeeded, and nothing CADET owns was left
  behind.
- **(b) metrics well formed** - the metrics, logs and provenance are present
  and parse.
- **(c) analysis** - the configured model answers a question about the run, and
  the answer matches a ground truth computed from the run's own files.
- **(d) dashboard renders** - the run view renders from the run directory.

A cell is `pass` when every check passes, `pass_partial` when one is blocked by
a missing dependency rather than failing, and `fail` when a check fails with the
dependency present.

## Verifying a change to code that runs inside a node

A cell places its roles as the config says, and a placed node runs the image's
own copy of CADET - not the working tree. Mounting the tree over it is not what
happens, so a change to anything a role executes is invisible to the matrix
until the image is rebuilt:

    docker build -f docker/Dockerfile -t cadet:dev \
        --build-arg GIT_SHA="$(git rev-parse HEAD)" .

A distributed cell needs it on the second machine too, which has no access to
this one's daemon:

    docker save cadet:dev | ssh <second machine> docker load

The image records the commit it was built from
(`org.opencontainers.image.revision`), and check (b) reads it into each run's
provenance, so a run says which code it actually ran:

    docker inspect cadet:dev --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'

Worth checking that label against `git rev-parse HEAD` before concluding
anything from a cell that places nodes in containers. A change tested only
against the host interpreter can pass its unit tests, pass on a `local` cell,
and have no effect whatever on a `docker` or `ssh_docker` one.

## One pass at a time

Run ids are derived from the resolved config, so two passes over the same cell
produce the same id - and therefore the same container names, the same network
name and the same staging paths. Running two passes at once makes them delete
each other's containers and networks mid-run, which surfaces as
`network ... already exists` or a staged config that has vanished. Neither
reads as contention. Run one pass at a time.

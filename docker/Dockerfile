# CADET unified image: the whole pipeline + the CARLA 0.9.10 client, in one env.
#
# This is a CARLA *client* image (it talks to a separate carla-sim container or a
# remote simulator); it does not run the simulator itself. Build from the repo
# root, with no preparation:
#   docker build -f docker/Dockerfile -t cadet:dev \
#       --build-arg GIT_SHA="$(git rev-parse HEAD)" .
#
# The build arg is not decoration. A role in a container runs the image's copy
# of the code, so an image built before a change does not have it - which looks
# like the change not working rather than like a stale image, and cost a day's
# measurement being wrong before the label existed. The commit is recorded in
# the image and the orchestrator warns when it does not match the repository it
# was started from. Omit the arg and the image records nothing, which the
# orchestrator reports as "cannot tell" rather than as agreement.
#
# Build args:
#   GIT_SHA                 the commit this image was built from, recorded as
#                           the OCI revision label. Unset means unknown.
#   INSTALL_ML=1  (default) install torch/torchvision/ultralytics (YOLO path).
#   INSTALL_ML=0            skip them for a small image that still runs the
#                           reference example and the test suite (stub model).
#   DOCKER_CLI_VERSION      the static docker *client* baked in, so a container
#                           from this image can drive the daemon of the machine
#                           it runs on (scripts/cadet does exactly that). A
#                           client only: no daemon and no privilege.
#
# Naming and paths are deliberately neutral: the image is built from a plain
# distribution base rather than from an upstream autonomy image, so it carries
# no inherited user, home directory or environment variable from one. Everything
# lives under /app and runs as the unprivileged `cadet` user.
#
# The CARLA client egg comes out of the official server image, as a build
# stage, so nothing has to be staged by hand and a clean clone builds with no
# preparation. CARLA_IMAGE is declared before the first FROM so it can be
# substituted into one; it is the same image sim.carla.image pins as the server.
ARG CARLA_IMAGE=carlasim/carla:0.9.10.1
FROM ${CARLA_IMAGE} AS carla

# The docker *client*, fetched in a stage of its own so that neither curl nor
# the 82 MB release tarball reaches the final image - only the 42 MB binary.
# This is a client and nothing else: it talks to a daemon over a socket the
# container is given, and no daemon, storage driver or privilege is added here.
# It is what lets `cadet run` start the simulator from inside a container, which
# is how the examples' `launcher: local` topology - the original's one container
# holding orchestrator and roles together - runs without a Python on the host
# that can import the CARLA client egg.
#
# Static rather than the distribution's package: the client has to match the
# daemon it is handed, not the base image's release, and a static binary is the
# only build that is a client on its own.
# The default belongs on the stage's own ARG: this is past the first FROM, so a
# declaration here would be scoped to the stage above and arrive empty in the
# one that reads it - which is a 404 for a tarball called docker-.tgz.
FROM ubuntu:20.04 AS dockercli
ARG DOCKER_CLI_VERSION=28.5.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
        "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
        -o /tmp/docker.tgz \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && rm /tmp/docker.tgz

# The base is Ubuntu 20.04, and the CARLA client is why. Python 3.8 is required
# (the 0.9.10 client exists only for cp37/cp38), but so is Ubuntu: the egg's
# native extension links against libjpeg.so.8, which is Ubuntu's soname. Debian
# ships libjpeg.so.62 and has no package providing .so.8, so on a Debian base
# the client can be baked in and still fail to import - which the build check
# below caught. The emulation image settled on the same base for the same
# reason; agreeing here keeps the two from differing in a way that only surfaces
# when an AV example needs a client.
FROM ubuntu:20.04

ARG INSTALL_ML=1
# ScenarioRunner's own dependency set, so the scenario driver can run in a
# container from this image. It is not optional in practice: the alternative was
# a shared virtualenv that had to be kept in step by hand, and a venv missing
# one of these fails a braking run at the scenario's first import, which reads
# as a startup timeout rather than a missing package.
ARG INSTALL_SIM=1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Repo root on the path: pip installs only the cadet* packages, and the
    # console script does not add the working directory, so --app-module
    # examples.* would not import. Extended further down to include the baked
    # CARLA client egg.
    PYTHONPATH=/app \
    # Headless pygame/OpenCV (no display in a container).
    SDL_AUDIODRIVER=dummy \
    SDL_VIDEODRIVER=dummy

# Runtime packages, in four groups:
#   * python3.8 and pip, since a distribution base brings no interpreter;
#   * OpenCV / pygame / Open3D need the GL and X stubs;
#   * the CARLA client's native extension (libcarla) links against libpng,
#     libjpeg-turbo8 and libtiff5 - without them `import carla` fails with a
#     bare "lib*.so.N: cannot open shared object file". Each was found by the
#     build check below naming it, one at a time, rather than by guessing: the
#     sonames are base-specific (Ubuntu's libjpeg.so.8 and libtiff.so.5 have no
#     Debian equivalent), so a guessed list breaks the build on one base for a
#     library the other needed;
#   * iproute2 for a node whose link uses network.mode netem: it installs its
#     own qdisc with tc, and needs NET_ADMIN at run time to do so;
#   * openssh-client for an orchestrator that runs in this image and places a
#     role on another machine. Both remote launchers front every command with
#     `ssh`, and without it the run failed with `No such file or directory:
#     'ssh'` from the first check it made of the other machine.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3-pip python3.8-distutils \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libpng16-16 libjpeg-turbo8 libtiff5 \
        iproute2 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python3

WORKDIR /app

# Install the config/CLI layer first (small, cache-friendly), then the full
# runtime deps only when building the ML image.
COPY pyproject.toml requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir pydantic==2.10.6 PyYAML==6.0.3 \
    && python -m pip install --no-cache-dir numpy==1.24.4 opencv-python-headless==4.9.0.80 \
        pygame==2.5.2
# The CARLA client egg, baked in at a fixed path.
#
# It cannot come from PyPI: version 0.9.10 was never published there (the index
# offers 0.9.3-0.9.5 and 0.9.12 onward). This was once a
# `pip install carla==0.9.10 || echo "WARN: ..."`, which failed on every build
# and carried on, so the image claimed a client it had never contained.
#
# It is copied out of the official server image above rather than staged by
# hand, which it used to be: that meant a clean clone could not build until
# someone found a PythonAPI tree and put a 23 MB binary in the context. The
# copy also gives something staging could not - the client comes from the *same*
# image the config pins as the server, so client/server skew is structurally
# impossible here rather than merely checked for afterwards.
#
# A run needing a different egg - an older image that keeps its PythonAPI
# elsewhere, say - sets sim.carla_client_egg_host_path, and that host directory
# is mounted over CARLA_EGG_PATH at run time.
# Must match cadet.sim.versions.CONTAINER_CARLA_EGG_DIR.
ENV CARLA_EGG_PATH=/opt/carla/PythonAPI
COPY --from=carla /home/carla/PythonAPI/carla/dist/ ${CARLA_EGG_PATH}/
# The egg goes on sys.path through a .pth file in site-packages rather than a
# literal PYTHONPATH entry, so whatever egg was staged is the one imported and
# the image does not hard-code a version in a second place. The import is then
# verified: an image that cannot import the client it baked is broken, and this
# is where that is cheapest to discover.
RUN set -e; \
    egg="$(ls ${CARLA_EGG_PATH}/carla-*-py3.7-*.egg 2>/dev/null | head -1)"; \
    if [ -z "$egg" ]; then \
        echo "ERROR: no CARLA py3.7 client egg copied from the server image." >&2; \
        echo "       The official image may have moved its PythonAPI; check" >&2; \
        echo "       /home/carla/PythonAPI/carla/dist/ in the CARLA_IMAGE." >&2; \
        exit 1; \
    fi; \
    site="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    printf '%s\n' "$egg" > "$site/carla-client-egg.pth"; \
    python -c 'import carla; print("carla client ok:", carla.__file__)'
RUN if [ "$INSTALL_ML" = "1" ]; then \
        python -m pip install --no-cache-dir \
            torch==2.4.1 torchvision==0.19.1 ultralytics==8.3.28 filterpy==1.4.5 \
            opencv-python==4.9.0.80 ; \
    fi
# Pinned to the same versions as the emulation image, so a scenario behaves the
# same whichever image hosts it.
RUN if [ "$INSTALL_SIM" = "1" ]; then \
        python -m pip install --no-cache-dir \
            py-trees==0.8.3 xmlschema==1.0.18 \
            ephem==4.2.1 tabulate==0.9.0 psutil==7.2.2 six==1.14.0 \
            Shapely==1.7.0 \
            'networkx==3.1' ; \
    fi

# The docker client from its fetch stage. Before the app code, so an edit to
# the repository does not invalidate it.
COPY --from=dockercli /tmp/docker/docker /usr/local/bin/docker

# App code last so edits don't bust the dependency layers.
COPY . .
RUN python -m pip install --no-cache-dir -e . --no-deps

# The commit this image carries, under the standard OCI label so anything can
# read it, not only CADET. Declared after COPY so it never invalidates a
# dependency layer: a label is metadata and re-labelling is not a rebuild.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
# The same value inside the image, because the label is readable only by asking
# the daemon and a process in the container cannot. There is no git here, so an
# orchestrator that runs in this image has no other way to say which code it
# carries, and a run record with no provenance at all is the worse answer.
ENV CADET_IMAGE_REVISION="${GIT_SHA}"

# An unprivileged user with a neutral name, owning what it needs to write.
# A role process has no reason to be root.
#
# Two things still need root and must override this with `--user root`:
#   * a node whose link uses `network.mode: netem`, because it installs its own
#     qdisc with tc (and also needs NET_ADMIN);
#   * the emulation image, which is a separate Dockerfile and stays root by
#     design because it drives netlink and the OVS datapath.
RUN useradd --create-home --shell /bin/bash cadet \
    && mkdir -p /app/outputs \
    && chown -R cadet:cadet /app
USER cadet

# Default: show CLI help. Compose / `docker run` override with a real command.
# The console script lands in ~/.local/bin for a pip --user install and in
# /usr/local/bin otherwise; both are on PATH for the cadet user.
ENV PATH=/home/cadet/.local/bin:$PATH
ENTRYPOINT ["cadet"]
CMD ["--help"]

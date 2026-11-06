# CADET network-emulation image: Mininet-WiFi + Open vSwitch + the cadet package.
#
# What this is for:
#     Hosts the `netwaggle` link backend. A container from this image builds a
#     Mininet-WiFi topology and runs CADET role processes inside its emulated
#     hosts, so link traffic crosses a real emulated network while channels
#     bypass it on a second, unshaped interface.
#
# What it contains:
#     An Ubuntu base matching the lab kernel series; Open vSwitch (ovsdb-server
#     plus ovs-vswitchd, started by hand since the image has no init); Mininet
#     and Mininet-WiFi, both pinned to release tags; the wireless userspace
#     Mininet-WiFi shells out to (hostapd, wpa_supplicant, iw, hwsim_mgmt);
#     measurement tools (ping, iperf3, tcpdump); and the cadet package itself.
#
# How it fits into CADET:
#     Built as `cadet-netwaggle:dev`. The orchestrator's docker launcher starts
#     one container from this image per run whose config has any link with
#     `network.mode: netwaggle`, then launches role processes inside the
#     topology's hosts. The separate `docker/Dockerfile` stays the plain CADET
#     client image: only this one needs privileges and kernel modules, and the
#     client image must not carry them.
#
# How to configure it:
#     Build args pin the emulator versions — MININET_TAG, MININET_WIFI_TAG — and
#     INSTALL_ML selects whether the detection stack is installed (0 keeps the
#     image small and still runs the reference example and the test suite).
#
#     Build from the repo root:
#         docker build -f docker/netwaggle.Dockerfile -t cadet-netwaggle:dev .
#
#     Run with its own network namespace (never host networking), privileged so
#     it can drive netlink and the OVS datapath, and with the host's modules
#     mounted read-only so openvswitch, sch_netem, ifb and mac80211_hwsim can be
#     loaded:
#         docker run -d --name cadet-netwaggle --privileged \
#             -v /lib/modules:/lib/modules:ro cadet-netwaggle:dev sleep infinity
#
#     For an AV example, additionally mount what must not be baked in, and put
#     the CARLA client on PYTHONPATH (the egg extracts at import, so the cache
#     directory has to be writable):
#         -v <models>:/models:ro
#         -v <scenario_runner>:/opt/scenario_runner:ro
#         -v <carla PythonAPI>:/opt/carla:ro
#         -e PYTHONPATH=/app:/opt/scenario_runner:/opt/carla/carla/dist/<egg>
#         --gpus device=<n> -e CUDA_VISIBLE_DEVICES=<n>
#
#     Note that mac80211_hwsim is a global kernel module: loading it from this
#     container loads it in the host kernel. The radios it creates are moved into
#     the container's namespaces, but the module itself is shared.
#
#     Python is 3.8 by design, not by accident. The CARLA 0.9.10 client ships as
#     a py3.7 egg (and PyPI wheels only up to cp38); it loads under 3.8 but not
#     under 3.10, so a newer base would leave the image unable to run any CARLA
#     example. 3.8 also matches the repo's own target (pyproject: py38).
FROM ubuntu:20.04

# Mininet-WiFi release tag, and the Mininet release it is built against. Upstream
# installs Mininet from master; pinning a tag keeps the image reproducible.
ARG MININET_WIFI_TAG=v2.4.3
ARG MININET_TAG=2.3.1b4
ARG INSTALL_ML=0
# ScenarioRunner's dependencies, so an app's sim hook can run in this container.
ARG INSTALL_SIM=0

# PIP_DEFAULT_TIMEOUT/PIP_RETRIES: the detection stack is hundreds of megabytes
# in a handful of wheels, and pip's default 15 s socket timeout is short for one
# of those on a busy link - the download fails most of the way through and the
# whole layer is lost.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# Build toolchain, Mininet core dependencies, the wireless userspace, Open
# vSwitch, and the measurement tools the feasibility checks and the accuracy
# tables rely on.
#
# Deviation from upstream's util/install.sh: that script builds hostapd,
# wpa_supplicant and iw from source (its bundled hostap submodule plus a
# kernel.org clone). This image uses the distribution packages instead, which
# cover AP and station association and avoid two slow external git fetches.
# Features that need the patched hostap (WPS, some EAP modes) are therefore not
# available here. wmediumd IS built from source below: it has no distribution
# package and the wireless medium is unusable without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git make gcc pkg-config help2man patch sudo \
        python3 python3-dev python3-pip python3-setuptools python3-pexpect \
        python3-tk \
        openvswitch-switch openvswitch-common \
        hostapd wpasupplicant iw wireless-tools rfkill \
        python3-shapely libgeos-dev \
        libjpeg-turbo8 libtiff5 libpng16-16 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libnl-3-dev libnl-genl-3-dev libssl-dev libevent-dev libdbus-1-dev \
        iproute2 iputils-ping iperf iperf3 tcpdump net-tools ethtool \
        socat psmisc procps kmod iptables \
    && rm -rf /var/lib/apt/lists/*

# numpy and matplotlib first: Mininet-WiFi's setup.py imports mn_wifi.net, which
# reaches matplotlib through its mobility plotting, so the install fails without
# it. Pinned to the unified environment's versions; the distribution's numpy is
# too old for matplotlib and scipy, which is why these come from pip.
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir numpy==1.24.4 matplotlib==3.7.5

# Mininet core (pinned). `make install` builds mnexec, installs the manpages and
# runs setup.py, which is what Mininet-WiFi expects to find already present.
RUN git clone --depth 1 --branch "${MININET_TAG}" \
        https://github.com/mininet/mininet.git /opt/mininet \
    && make -C /opt/mininet install PYTHON=python3

# Mininet-WiFi (pinned). Installed the same way; provides the mn_wifi package.
RUN git clone --depth 1 --branch "${MININET_WIFI_TAG}" \
        https://github.com/intrig-unicamp/mininet-wifi.git /opt/mininet-wifi \
    && make -C /opt/mininet-wifi install PYTHON=python3

# hwsim_mgmt: the helper Mininet-WiFi calls to create *named* mac80211_hwsim
# radios (mn_wifi/module.py). Without it mn_wifi falls back to anonymous radios,
# which breaks per-node interface naming.
RUN git clone --depth 1 \
        https://github.com/ramonfontes/mac80211_hwsim_mgmt.git /opt/hwsim_mgmt \
    && make -C /opt/hwsim_mgmt install

# wmediumd: the wireless medium daemon. Mininet-WiFi needs it for anything the
# propagation model drives — signal strength, interference, association as a
# function of position, mobility. Without it a wifi link degrades to a plain
# shared medium and the configured positions are ignored.
RUN apt-get update && apt-get install -y --no-install-recommends libconfig-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 -b mininet-wifi \
        https://github.com/ramonfontes/wmediumd.git /opt/wmediumd \
    && make -C /opt/wmediumd install

WORKDIR /app

# Config/CLI layer first (small, cache-friendly), then the optional ML stack.
COPY pyproject.toml requirements.txt ./
# pygame is not optional any more: cadet/roles/client_visualizer.py renders the
# original's per-tick window, which the braking apps switch on by default.
# Without it a braking client fails at setup.
# psutil is CADET's, not the simulator's: the cpu collector reads through it,
# and without it a run that asks for `metrics.collectors: [cpu]` fails at the
# role's start with "psutil is not installed" - which it did on every emulated
# acceptance cell, because the pin sat in the simulator's optional block.
RUN pip3 install --no-cache-dir pydantic==2.10.6 PyYAML==6.0.3 \
        opencv-python-headless==4.9.0.80 filterpy==1.4.5 pygame==2.5.2 pytest \
        psutil==7.2.2
# opencv-python is pinned here as well as headless above, and the reason is
# not redundancy: ultralytics declares `opencv-python` (a different
# distribution from `opencv-python-headless`), unpinned, so pip installs it
# alongside and `import cv2` resolves to whichever landed last. Measured in
# this image before the pin: opencv-python 5.0.0.93 beside the pinned headless
# 4.9.0.80, with cv2 reporting 5.0.0 - a major version the repository never
# asked for, in the library the braking app writes its annotated frames with.
# The wheel cache is a mount rather than a layer: it survives a failed build,
# so a retry installs what already arrived instead of fetching it again, and it
# is not committed into the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ML" = "1" ]; then \
        PIP_NO_CACHE_DIR=0 pip3 install torch==2.4.1 torchvision==0.19.1 \
            ultralytics==8.3.28 opencv-python==4.9.0.80 ; \
    fi

# ScenarioRunner's dependency set, fully pinned.
#
# Upstream's requirements.txt pins only some of these and leaves psutil, ephem,
# tabulate and six floating, so the image resolved whatever was current at build
# time and two builds of the "same" image could differ. Every version below is
# pinned at what this image resolved and what the lifecycle rows pass with.
#
# Two deliberate deviations from upstream's pins, both forced:
#   * networkx 3.1, not 2.2: 2.2 calls np.int, which numpy removed in 1.24, and
#     1.24 is what the detector stack here needs.
#   * Shapely 1.7.0, not 1.6.4.post2: upstream's pin has no wheel for this
#     Python and 1.7.0 is what builds. It was previously arriving transitively
#     and undeclared, which is worse than a stated deviation.
#
# ScenarioRunner and the CARLA PythonAPI are NOT copied in: they are external
# repositories, so they are mounted read-only at run time and put on PYTHONPATH.
RUN if [ "$INSTALL_SIM" = "1" ]; then \
        pip3 install --no-cache-dir \
            py-trees==0.8.3 xmlschema==1.0.18 \
            ephem==4.2.1 tabulate==0.9.0 six==1.14.0 \
            Shapely==1.7.0 \
            'networkx==3.1' ; \
    fi

# App code last so edits do not bust the dependency layers.
COPY . .
RUN pip3 install --no-cache-dir -e . --no-deps

# The commit this image carries, under the standard OCI label, and the same
# value in the environment because a process in the container cannot ask the
# daemon for a label. Declared after COPY so re-labelling is not a rebuild.
# Without it a run in this image records no provenance and the orchestrator
# cannot tell whether the image carries the code in the working tree.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
ENV CADET_IMAGE_REVISION="${GIT_SHA}"

# No init in the image: OVS and the emulator are started explicitly by the
# orchestrator (see cadet/network/netwaggle). Default to a shell so the container
# can be kept alive and driven with exec.
CMD ["/bin/bash"]

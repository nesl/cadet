Tutorial 4: Multi-Device Deployment
=====================================

.. note::

   This tutorial requires a GPU host with Docker.

.. note::

   All four launchers run without an opt-in: ``local``, ``docker``,
   ``ssh_docker``, and ``ssh_venv`` for a device that cannot hold a container
   image.

The docker launcher runs each node in its own container, named
``cadet-{run_id}-{node_name}``. The ssh_docker launcher wraps the docker
command with an SSH hop to a remote host.

docker launcher
---------------

Set ``launcher: docker`` on any node:

.. code-block:: yaml

   extends: configs/base.yaml

   experiment: {name: docker_test, app: reference, repetitions: 1}
   source:
     kind: synthetic
     count: 50
     width: 224
     height: 224
   nodes:
     - name: client
       host: localhost
       role: client
       device: cpu
       launcher: docker
       image: cadet:dev

     - name: server
       host: 127.0.0.1
       port: 15010
       role: server
       device: gpu
       gpu_index: 0
       launcher: docker
       image: cadet:dev

   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution: {kind: constant, value_ms: 30.0}
   app: {plugin: reference, params: {model_kind: stub}}

The orchestrator passes:

- ``--gpus device=0`` and ``-e CUDA_VISIBLE_DEVICES=0`` for nodes with
  ``gpu_index: 0``.
- ``--cap-add NET_ADMIN`` for nodes with a netem or netwaggle link.
- ``-v {outputs_dir}:/app/outputs`` to bind-mount the output directory.
- ``--name cadet-{run_id}-{node_name}`` as the container name.

Containers are removed with ``docker rm -f`` on teardown. Running
``cadet down`` also removes any ``cadet-*`` containers left by a crash.

ssh_docker launcher
-------------------

Set ``launcher: ssh_docker`` and say how the machine is reached. The fields are
the node's own, not an ``ssh_target`` string, so each one can be set or swept
independently:

.. code-block:: yaml

   nodes:
     - name: server
       host: 203.0.113.20
       port: 15010
       role: server
       device: gpu
       gpu_index: 1
       launcher: ssh_docker
       container: cadet:dev
       user: an-account
       ssh_identity: /path/to/the/orchestrators/private/key

The launcher wraps ``docker run ...`` with ``ssh -o BatchMode=yes -o
ConnectTimeout=10 -i <ssh_identity> an-account@203.0.113.20 docker run ...``.
``host`` is both the address the SSH goes to and the address the node's peers
connect to: the container joins that machine's own network namespace, because
whether a *published* port is reachable depends on that machine's forwarding
policy and nothing in a config can promise it.

Three things are true of a node reached this way, and each is a property of the
other machine rather than of this one:

* **the outputs are staged there and fetched back.** A bind mount names a path
  in the filesystem of the machine the container runs on, so this orchestrator's
  output directory means nothing there. CADET creates a per-run tree under that
  account's own home, mounts it, copies it into the local outputs root when the
  run ends, and removes it.
* **the run's config is sent, not named.** The role would otherwise resolve the
  config path against whatever copy of the repository the image was built with,
  and a config written or edited since is absent there. CADET stages the
  resolved config - after ``extends``, after the ``--set`` overrides and after
  the sweep is expanded to one run - and points the role at it. So the second
  machine needs no checkout: just a docker daemon its account can use, the
  image, and this orchestrator's public key in its ``authorized_keys``.
* **the container runs as that account's uid:gid**, read from the account
  itself, because this process's uid need not exist there and would not own the
  mount.

Run a two-machine experiment from a host account rather than from
``scripts/cadet``: the orchestrator authenticates as the invoking account with
that account's own key, and a key does not belong inside a container image or a
container's mounts.

A ``netem`` link whose *sending* node is reached this way is refused in
preflight. A qdisc needs the sending node's own network namespace and this
orchestrator has no handle on one on another machine; shape such a path from the
local end instead, which is a node with a namespace of its own.

Once the launchers pass ``cadet validate``, run the docker test config with:

.. run-tag:: gpu-host-only

.. code-block:: shell

   cadet run examples/docker_test/docker_test.yaml

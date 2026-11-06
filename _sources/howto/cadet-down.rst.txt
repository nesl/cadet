Recovery with cadet down
========================

``cadet down`` restores a clean state after a crash or interrupted run::

    cadet down

It performs these steps in order:

1. Kills all processes whose PID is in the state store, plus any process
   with ``cadet serve``, ``cadet client``, or ``cadet load`` in its command
   line.
2. Runs ``docker rm -f`` on all containers whose name starts with
   ``cadet-``.
3. Removes any Docker network named ``cadet-*``.
4. Removes emulation network namespaces, OVS bridges, and veth interfaces
   left by the netwaggle backend.
5. Removes root netem qdiscs on interfaces recorded in the state store.
6. Clears the state store.

When to use it
--------------

Run ``cadet down`` before starting a new experiment if:

- A previous run was interrupted with Ctrl-C, SIGKILL, or a system restart.
- ``cadet status`` shows stale pids.
- ``cadet run`` fails at preflight with "port already in use" and you
  cannot identify the holder with ``lsof``.

Foreign port holder
-------------------

If a port is in use by a process CADET did not start, preflight names the
holder in the error::

    PreflightError: port 15010 held by pid 12345 (not a CADET process)

CADET refuses to kill foreign processes. Resolve this manually::

    kill 12345

or change the ``port`` field on the server node to a free port.

Leftover netem state
--------------------

A netem qdisc left on an interface from a crashed run can cause unexpected
latency on the next run. ``cadet down`` removes qdiscs it recorded in the
state store. If the state store was lost (e.g. after a reboot), remove
qdiscs manually::

    tc qdisc del dev lo root

Replace ``lo`` with the interface name used by the experiment.

GPU selection
-------------

Set ``gpu_index`` on the server node to select a specific GPU:

.. code-block:: yaml

   nodes:
     - name: server
       host: 127.0.0.1
       port: 15010
       role: server
       device: gpu
       gpu_index: 1

Inside a container, also set ``CUDA_VISIBLE_DEVICES`` to the same index::

    docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=1 cadet:dev ...

When ``CUDA_VISIBLE_DEVICES=1``, CUDA sees only that GPU as device 0
inside the container, so the model loads onto the physical GPU at index 1.

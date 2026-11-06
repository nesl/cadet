Metrics Reference
=================

Two metric files are written for every run. Additional collectors can be
enabled via ``metrics.collectors`` in the config.

See :doc:`../reference/output-format` for the output directory layout and
:ref:`orch-lifecycle` for the orchestrator lifecycle that produces them.

----

Reference app: frame latency log
----------------------------------

**File:** ``{run_id}_latency_location_log.csv``

Written by the reference app client on experiment completion. One row per
frame sent during the run.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Column
     - Type
     - Description
   * - ``frame_id``
     - int
     - Sequential frame number, starting at 0.
   * - ``latency_s``
     - float (s)
     - Round-trip time from frame send to response receipt, in seconds.
   * - ``server_processing_time``
     - float (s)
     - Time spent on model inference for this frame, in seconds.
   * - ``result``
     - dict/str
     - Model output returned by the server for this frame.

Example (8-frame synthetic run)::

    frame_id,latency_s,server_processing_time,result
    0,0.013381,4.9e-05,"{'mean': 0.0, 'shape': [48, 64, 3]}"
    1,0.019557,2.2e-05,"{'mean': 1.0, 'shape': [48, 64, 3]}"

----

Braking app: frame latency log
--------------------------------

**File:** ``{run_id}_latency_location_log.csv``

Written by the braking app client. One row per frame captured and sent
during the run. Columns include CARLA world coordinates and braking state.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Column
     - Type
     - Description
   * - ``frame_id``
     - int
     - Sequential frame number, starting at 0.
   * - ``frame_capture_time``
     - float (s)
     - Simulation time when the frame was captured.
   * - ``frame_capture_loc``
     - tuple
     - Ego vehicle location (x, y, z) at capture time (CARLA world coordinates).
   * - ``frame_send_time``
     - float (s)
     - Simulation time when the frame was sent to the server.
   * - ``frame_send_loc``
     - tuple
     - Ego vehicle location at send time.
   * - ``server_processing_time``
     - float (s)
     - Time the server spent on model inference for this frame.
   * - ``server_response_receive_time``
     - float (s)
     - Simulation time when the client received the server's response for a
       brake-triggering frame.
   * - ``server_response_loc``
     - tuple
     - Ego vehicle location when the brake response was received.
   * - ``brake_issued``
     - bool
     - ``True`` for the frame that triggered a brake command.
   * - ``stopping_time``
     - float (s)
     - Simulation time the vehicle came to a full stop (braked frames only).
   * - ``stopping_loc``
     - tuple
     - Location of the full stop.
   * - ``vehicle_speed_mph``
     - float
     - Ego speed at frame capture time, in mph.
   * - ``run_id``
     - str
     - Run identifier (see :doc:`../explanation/reproducibility`).
   * - ``rep``
     - int
     - Repetition index within the sweep cell.
   * - ``brake_reaction_latency_s``
     - float (s)
     - Wall-clock round-trip time from frame capture to brake response
       receipt, for the brake-triggering frame. ``None`` for all other
       frames. Model warm-up (``models.warmup: true``, the default) runs
       one inference before the experiment to avoid JIT-compilation cost
       on the first real frame.

----

Link report
-----------

**File:** ``link_report.json``

Written by the client role at the end of every run. Records the configured
network parameters and per-message latency statistics.

Top-level structure::

    {
      "experiment": "ref_synth",
      "app": "reference",
      "configured_links": [],
      "measured": {
        "reply_policy": "per-message",
        "messages_sent": 8,
        "messages_received": 8,
        "unanswered": 0,
        "unanswered_pct": 0.0,
        "latency_ms": {
          "samples": 8,
          "p50": 19.39,
          "p99": 19.56,
          "mean": 18.63,
          "min": 13.38,
          "max": 19.56
        }
      }
    }

``configured_links`` lists each link's ``network`` block from the resolved
config. ``measured`` covers all messages across the run.

``unanswered`` counts frames the server never replied to. With ``mode: delay``
and ``loss > 0``, messages are dropped at the application layer, so
``unanswered`` can be non-zero. With ``mode: netem`` or ``mode: netwaggle``,
TCP retransmits lost packets, so frames arrive late but ``unanswered`` stays 0.

----

Queue collector
---------------

**Enabled by:** ``metrics.collectors: [queue]``

Samples the server's request queue depth at a fixed interval during the run.

**File:** ``queue_log.csv``

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Column
     - Type
     - Description
   * - ``t``
     - float (s)
     - Wall-clock timestamp of the sample.
   * - ``queue_depth``
     - int
     - Number of requests waiting for model inference at sample time.

----

The other collectors
--------------------

Each runs on the node it measures and writes its own per-node CSV:

- ``gpu``: utilisation and memory, sampled every 10 ms by default.
- ``cpu``: host and per-process utilisation and memory, every 100 ms.
- ``bandwidth``: payload bytes per link per direction, every 100 ms.
- ``energy``: power draw, from the reader the node's hardware offers, with
  every row recording which reader produced it.
- ``clock_offset``: each node's clock against the orchestrator's, measured
  twice per run, with the uncertainty of the estimate. It records; it corrects
  no timestamp.

Why ``clock_offset`` is measured per run, and not once for the lab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two clocks differ in **rate**, not only in offset. Measured on the pair of
machines in use here, the rates differ by about 2.5 parts per million, recovered
three independent ways: the offset moved 9.8 to 11.2 microseconds inside a
single 4.2 second run, 114 microseconds across four runs spanning 39 seconds,
and the two time daemons' own frequency corrections differ by 2.3 parts per
million. An offset measured once for the lab would therefore be wrong by more
than its own uncertainty within a minute of being measured.

The estimate is recorded with its half-width, its method and its sign
convention, and **nothing is corrected**: on this pair the offset is
-0.278 ± 0.063 ms, so subtracting it from a cross-node interval of a few tenths
of a millisecond yields a number whose sign is not determined, while subtracting
it from a 60 ms shaped-link measurement is a resolvable half-percent. The record
cannot tell which a reader intends, so it reports and leaves the choice.

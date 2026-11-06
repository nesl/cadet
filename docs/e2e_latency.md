# The `brake_reaction_latency_s` metric (formerly `e2e_latency_s`)

## What it pairs
The open-loop braking server sends one reply per run: the brake
command. That reply carries the `frame_id` of the frame whose detection
pushed the consecutive-detection counter to the threshold (the *decisive* frame).

`brake_reaction_latency_s` = (wall-clock time the client receives the brake)
− (wall-clock capture time of that decisive frame).

So it pairs one send (the decisive frame) with one reply (the brake). It
is a single-sample, task-level "reaction latency", not a per-frame stream metric.
Because the server never acks non-brake frames, no per-frame round-trip exists to
average.

## Why it is ~1.4 s (not tens of ms)
Measured on this host (yolo11x, A6000): the **first** `model.predict()` takes
**~0.99 s** (CUDA/model warmup — kernel compilation and caching), while steady
state is **~0.017 s**. The client streams frames at camera rate while the
server's first inference is still running, so a queue builds. The decisive frame
is therefore processed ~1 s (warmup) + queue drain after it was captured, giving
a reaction latency around 1.3–1.5 s.

This warmup inflation is a property of the simulation harness (and is present in
the original client too), not of the network path. It is **not** removed by
warming up the server, because doing so would change when detection starts and
thus break parity with the original. The directly comparable, warmup-insensitive
latency figure used in the parity tables is `server_processing_time` (the
per-frame server-side duration), reported separately.

## Original-vs-new clock note
The original CSV mixes clocks (`frame_capture_time` in wall-clock,
`server_response_receive_time` in sim-time), so a clean reaction latency is not
recoverable from the baseline CSV. The new column fixes this by differencing two
wall-clock stamps.

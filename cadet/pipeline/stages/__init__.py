"""The CADET stage library: ported, placeable pipeline stages.

What this is for:
    Concrete :class:`cadet.pipeline.stage.Stage` implementations, ported (not
    rewritten) from the flat example apps and the vendored AV pipeline. Importing a
    module here registers its stages under their ``impl`` names.

What it contains (grows per phase):
    - :mod:`cadet.pipeline.stages.braking` — the braking subset: ``yolo`` detector,
      ``roi_consecutive`` decision, ``brake_controller`` control.
    - :mod:`cadet.pipeline.stages.v2v` — the V2V subset: ``yolo_v2v`` lead-range
      detector and ``v2v_ttc_gate`` brake decision (control reuses ``brake_controller``).
    - :mod:`cadet.pipeline.stages.v2i` — the V2I subset: ``yolo_v2i`` per-feed detector
      and the ``v2i_ego_gate`` / ``v2i_fusion_gate`` brake decisions (control reuses
      ``brake_controller``).

How it fits into CADET:
    A pipeline config names stages by ``impl``; the builder resolves them against
    the registry. Example packages import the stage modules they use (and add their
    own custom stages) before a run.

How to configure it:
    Per-stage ``params`` come from the config's ``pipeline.stages`` entries.
"""

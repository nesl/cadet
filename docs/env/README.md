# Recorded environment

- Python (both envs): 3.8.10
- CARLA: 0.9.10 PythonAPI egg (carla-0.9.10-py3.7-linux-x86_64.egg)
  under dependencies/CARLA_0.9.10.1/PythonAPI/carla/dist/

## Unified environment (current target)
- `requirements.lock.txt` — the full pinned set for a single venv that runs the
  entire pipeline **and** the YOLO detection path (torch 2.4.1 + cu121,
  ultralytics 8.3.28). All 24 pipeline modules import under it.
- Extra tracker dependencies: `scikit-image` is required (SORT); `lapsolver` is
  an optional extra (DeepSORT only, imported lazily).

See repo-root `requirements.txt` for the curated, import-driven direct deps.

## Earlier split-environment probes (provenance)
- `pip-freeze-system-python.txt` — original full pipeline + client (torch 1.12.0).
- `pip-freeze-yolo11-venv.txt` — original detection/YOLO venv (torch 2.4.1).

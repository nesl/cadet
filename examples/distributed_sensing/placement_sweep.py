"""One-command placement sweep: on-device vs edge vs cloud, across access qualities.

What this is for:
    The three placements the example compares - detector on the sensor, on the
    gateway, in the cloud - do not fit one sweep file, because the cloud placement
    changes the topology (the gateway becomes a waypoint). So the full comparison
    is two `cadet sweep` runs: `edge_<N>.yaml` (placements sensor and gateway x
    three access qualities) and `cloud_<N>.yaml` (cloud x the same three). This
    script runs both, reduces every run into one `summary.csv`, prints the table,
    and draws the crossover plot - latency and the task metric against access
    quality, one line per placement - in a single command.

What it writes (these columns are the contract; COLUMNS below is the list):
    outputs/sensing_placement_<N>/
      summary.csv    one row per run across both sweeps (the file a dashboard reads)
      summary.txt    the same rows as the fixed-width console table
      crossover.png  latency (left) and task metric (right) vs access quality

How to run it:
    python examples/distributed_sensing/placement_sweep.py --sensors 5
    python examples/distributed_sensing/placement_sweep.py --sensors 2 --no-run   # reuse outputs

How it fits into CADET:
    It shells out to `cadet sweep` (the framework orchestrator) and reads only the
    CSVs those runs wrote under outputs/. It writes nothing into the repository.
"""

import argparse
import csv
import os
import subprocess
import sys

# Sibling modules, imported the way summarise.py is run (its own dir on sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Placement is the categorical identity of the plot: fixed order, fixed colour,
# fixed marker, so a placement keeps its appearance across every figure. Colours
# are the data-viz reference palette's categorical slots 1-3 (CVD-validated as a
# set); the marker is a second, redundant encoding so identity never rests on
# colour alone.
PLACEMENTS = ["sensor", "gateway", "cloud"]
PLACEMENT_STYLE = {
    "sensor": {"color": "#2a78d6", "marker": "o"},   # slot 1, blue
    "gateway": {"color": "#eb6834", "marker": "s"},  # slot 2, orange
    "cloud": {"color": "#1baf7a", "marker": "^"},    # slot 3, aqua
}

# Access qualities, ordered worst-last so the x-axis runs good -> severe. The
# labels come from (p50_ms, loss); a cell matching none is "other".
QUALITY_ORDER = ["good", "contested", "severe", "other"]
QUALITY_TABLE = {
    (8.0, 0.0): "good",
    (35.0, 0.02): "contested",
    (90.0, 0.08): "severe",
}

# summary.csv columns, in order. This list is the contract: anything reading the
# file reads by name, so a rename here is a breaking change and belongs in the
# same commit as its readers.
SUMMARY_COLUMNS = [
    "sweep", "run", "run_dir", "placement", "sensors", "access_quality",
    "access_p50_ms", "access_p99_ms", "access_loss", "access_bandwidth_mbps",
    "answered", "lost", "e2e_p50_ms", "e2e_p99_ms", "aoi_p50_ms", "aoi_p99_ms",
    "frames_with_box", "uplink_kb",
]


def quality_label(p50, loss):
    """Name the access quality from its (p50_ms, loss), or 'other'."""
    if p50 is None or loss is None:
        return "other"
    return QUALITY_TABLE.get((float(p50), float(loss)), "other")


def run_sweep(config, extra_set):
    """Run one `cadet sweep`, inheriting this interpreter and the app module."""
    cmd = [
        sys.executable, "-m", "cadet", "sweep", config,
        "--app-module", "examples.distributed_sensing.app",
    ]
    for item in extra_set:
        cmd += ["--set", item]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def outputs_root(extra_set):
    """Where the runs were written, honouring an ``outputs.root`` override.

    The sweeps are given whatever ``--set`` this wrapper was given, so a run
    told to write somewhere else does. Aggregation has to look where the runs
    actually are: reading the default tree regardless reported "no runs found"
    for a sweep that had just finished successfully somewhere else.
    """
    root = None
    for item in extra_set or []:
        key, _, value = item.partition("=")
        if key.strip() == "outputs.root" and value:
            root = value.strip()
    if not root:
        return os.path.join(REPO_ROOT, "outputs")
    return root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)


def run_dirs_under(experiment, root=None):
    """Every run directory of one sweep's output tree, sorted."""
    import glob

    base = root or os.path.join(REPO_ROOT, "outputs")
    pattern = os.path.join(base, experiment, "distributed_sensing", "*-r*")
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def rows_for(experiment, dashboard_root=None, root=None):
    """Reduce every run of one sweep into a summary row.

    When ``dashboard_root`` is given, also assemble a unified run tree under it:
    each run directory is copied to ``<dashboard_root>/<run_id>/`` with the
    dashboard's per-run ``sensing_summary.json`` sidecar written in (first numeric
    entry = the headline metric, unit from the key suffix). One
    ``cadet dashboard <dashboard_root>`` then shows every placement x access cell
    of the sweep, because the two sweeps' runs - which land under separate
    ``outputs/sensing_{edge,cloud}_N`` trees - are gathered into one. The combined
    ``summary.csv`` and ``crossover.png`` the wrapper writes are for a person; the
    sidecars and the gathered tree are for the dashboard. ``run_dir`` in the row
    points at the original run, not the copy, so provenance stays with the sweep.
    """
    import shutil

    rows = []
    for run_dir in run_dirs_under(experiment, root):
        row = summarise.summarise(run_dir)
        row["run_dir"] = os.path.relpath(run_dir, REPO_ROOT)
        row["access_quality"] = quality_label(row.get("access_p50_ms"), row.get("access_loss"))
        if dashboard_root is not None:
            dest = os.path.join(dashboard_root, os.path.basename(run_dir))
            shutil.copytree(run_dir, dest, dirs_exist_ok=True)
            summarise.write_summary(dest, row)
        else:
            summarise.write_summary(run_dir, row)
        rows.append(row)
    return rows


def write_summary_csv(rows, path):
    """Write summary.csv with exactly the documented columns, by name."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in SUMMARY_COLUMNS})


def _cell(value):
    """Render a CSV cell: blank-safe, no scientific notation for small losses."""
    if value is None:
        return "n/a"
    return value


def plot_crossover(rows, path, sensors, backend):
    """Draw latency (left) and task metric (right) vs access quality, by placement.

    Two panels rather than two y-axes on one: the measures have different scales
    and units, so they get one chart each (small multiples), sharing the x. The
    left panel is the crossover the example exists to show - where the placements
    trade places as the access link degrades.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [q for q in QUALITY_ORDER if any(r["access_quality"] == q for r in rows)]
    x_of = {q: i for i, q in enumerate(present)}

    def series(placement, column):
        """(x, y) for one placement over the present qualities, skipping gaps."""
        xs, ys = [], []
        for q in present:
            match = [
                r for r in rows
                if r["placement"] == placement and r["access_quality"] == q
                and r.get(column) is not None
            ]
            if match:
                xs.append(x_of[q])
                ys.append(float(match[0][column]))
        return xs, ys

    text_primary, text_muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    fig, (ax_lat, ax_task) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    panels = [
        # (axis, column, title, subtitle, direct-label the lines?)
        (ax_lat, "e2e_p50_ms", "End-to-end latency p50 (ms)", "lower latency is better", True),
        (ax_task, "frames_with_box", "Detection rate (task metric)", "fraction of frames with a box", False),
    ]
    placements_present = [p for p in PLACEMENTS if any(r["placement"] == p for r in rows)]

    for ax, column, title, subtitle, label_lines in panels:
        for placement in placements_present:
            xs, ys = series(placement, column)
            if not xs:
                continue
            style = PLACEMENT_STYLE[placement]
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    linewidth=2.0, markersize=7, label=placement, zorder=3)
            # Direct label at the last point, but only where the series separate;
            # on the near-flat task panel the legend carries identity instead, so
            # the labels do not pile on top of each other.
            if label_lines:
                ax.annotate(placement, (xs[-1], ys[-1]), color=style["color"],
                            fontsize=9, fontweight="bold", xytext=(6, 0),
                            textcoords="offset points", va="center")
        ax.set_title(title, color=text_primary, fontsize=11, pad=10)
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, color=text_muted,
                fontsize=8.5, va="bottom")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)
        ax.grid(True, axis="y", color=grid, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=text_muted, labelsize=9)

    ax_lat.set_xlabel("access-link quality", color=text_muted, fontsize=9)
    ax_task.set_xlabel("access-link quality", color=text_muted, fontsize=9)
    handles, labels = ax_lat.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
               ncol=len(labels), bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(
        "distributed sensing detector placement: {} sensors, {} backend".format(sensors, backend),
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=text_primary,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensors", type=int, default=5, choices=[2, 5, 10],
                        help="sensor count; selects edge_<N>.yaml and cloud_<N>.yaml")
    parser.add_argument("--no-run", action="store_true",
                        help="reuse existing outputs instead of running the sweeps")
    parser.add_argument("--set", action="append", default=[],
                        help="passed through to both cadet sweeps (repeatable)")
    args = parser.parse_args(argv)

    n = args.sensors
    edge, cloud = "sensing_edge_{}".format(n), "sensing_cloud_{}".format(n)
    if not args.no_run:
        run_sweep("examples/distributed_sensing/edge_{}.yaml".format(n), args.set)
        run_sweep("examples/distributed_sensing/cloud_{}.yaml".format(n), args.set)

    root = outputs_root(args.set)
    out_dir = os.path.join(root, "sensing_placement_{}".format(n))
    os.makedirs(out_dir, exist_ok=True)
    rows = rows_for(edge, out_dir, root) + rows_for(cloud, out_dir, root)
    if not rows:
        print("no runs found; drop --no-run to run the sweeps first", file=sys.stderr)
        return 1
    # Stable order: placement, then quality severity.
    rows.sort(key=lambda r: (PLACEMENTS.index(r["placement"]) if r["placement"] in PLACEMENTS else 9,
                             QUALITY_ORDER.index(r["access_quality"])))

    table = summarise.format_rows(rows)
    print(table)

    csv_path = os.path.join(out_dir, "summary.csv")
    txt_path = os.path.join(out_dir, "summary.txt")
    png_path = os.path.join(out_dir, "crossover.png")
    write_summary_csv(rows, csv_path)
    with open(txt_path, "w") as handle:
        handle.write(table + "\n")
    try:
        plot_crossover(rows, png_path, n, "delay")  # the shipped configs use delay
        plotted = os.path.relpath(png_path, REPO_ROOT)
    except Exception as exc:  # a missing plotting stack must not lose the table
        plotted = "skipped ({})".format(exc)

    rel_out = os.path.relpath(out_dir, REPO_ROOT)
    print()
    print("wrote {}".format(os.path.relpath(csv_path, REPO_ROOT)))
    print("wrote {}".format(os.path.relpath(txt_path, REPO_ROOT)))
    print("plot  {}".format(plotted))
    print("runs  {}/<run_id>/ (per-run sensing_summary.json sidecars)".format(rel_out))
    print("view  cadet dashboard {}".format(rel_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

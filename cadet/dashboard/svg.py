"""Emit the dashboard's plots as SVG, with nothing fetched at page load.

What this is for:
    The page is demonstrated on a laptop that may have no network, so a plot
    that calls out to a charting library is a plot that renders nothing at the
    meeting. These functions build the SVG markup themselves: axes, ticks,
    marks, legend and the per-mark hover titles, as a string the page embeds
    inline. No script runs and no stylesheet or font is fetched.

What it contains:
    - :data:`SERIES_COLOURS`: the categorical slots, assigned in fixed order.
    - :func:`nice_ticks`, :func:`Scale`: the axis arithmetic.
    - :func:`crossover_plot`: the metric against the swept axis, one line per
      placement, which is the sweep view's plot.
    - :func:`bar_panel`: a horizontal bar per named value (per-stage latency).
    - :func:`series_panel`: one quantity over time (the device view's panels).
    - :func:`empty_plot`: the message a panel shows instead of failing.

How it fits into CADET:
    Called by :mod:`cadet.dashboard.render`, which embeds the returned markup in
    the page. Takes plain numbers, so it knows nothing about run directories.

    Colour is addressed through CSS custom properties the page defines
    (``--series-N``, ``--viz-grid``, ``--viz-ink``), which is what makes one
    emitter serve both the light and the dark theme: the page redefines the
    properties and the same markup restyles. Text never wears a series colour -
    the mark beside it carries the identity.

How to configure it:
    Nothing. Sizes are module constants; a caller passes data and a title.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Categorical slots, assigned in this fixed order and never cycled. Addressed as
# CSS custom properties rather than hex so the page's dark theme restyles the
# same markup; the hex values live in the page's stylesheet.
#
# A ninth series is not a generated colour: callers fold the tail into one
# "other" series instead, so identity never depends on a hue nobody chose.
SERIES_COLOURS = tuple("var(--series-{})".format(i) for i in range(1, 9))
MAX_SERIES = len(SERIES_COLOURS)

# Plot geometry. The margins are what the axis labels need, not decoration: a
# tick label that overflows its margin is the one layout fault this emitter can
# produce, so the margins are generous on the labelled sides.
PLOT_WIDTH = 720
PLOT_HEIGHT = 320
MARGIN_LEFT = 72
MARGIN_RIGHT = 24
MARGIN_TOP = 16
MARGIN_BOTTOM = 56

# Mark specs: thin lines, markers big enough to hit, a surface ring on an
# overlapping marker so two series crossing stay two series.
LINE_WIDTH = 2
MARKER_RADIUS = 4.5
MARKER_RING = 2
BAR_RADIUS = 4


def escape(text: Any) -> str:
    """Escape text for both element content and a double-quoted attribute.

    Everything plotted here comes out of a run directory - a node name, a link
    name, a config value - so none of it is trusted to be markup-safe.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt(value: Optional[float], places: int = 3) -> str:
    """Format a number for a label: short, and a dash when there is none.

    Four digits is a plain number, not an exponent. The threshold for
    scientific notation used to be a thousand, which turned every latency over
    a second into ``1.16e+03`` - and a page of millisecond figures is exactly
    where four digits are ordinary. So a magnitude of a thousand up to a
    million drops its decimals (``1156``) and keeps reading as a measurement;
    only genuinely large or genuinely tiny values become exponents, where an
    exponent is the readable form.
    """
    if value is None:
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 0.001:
        return "{:.2e}".format(value)
    if magnitude >= 1000:
        return "{:.0f}".format(value)
    text = "{:.{}f}".format(value, places).rstrip("0").rstrip(".")
    return text or "0"


def nice_ticks(low: float, high: float, count: int = 5) -> List[float]:
    """Round tick positions covering ``[low, high]``.

    A plain linear division puts ticks on values nobody would choose (0.133,
    0.267); stepping by 1, 2, 2.5 or 5 times a power of ten puts them on values
    a reader can compare against. Returns a single tick for a degenerate range,
    which is what a series with one distinct value has.
    """
    if not (high > low):
        return [low]
    span = high - low
    raw = span / max(1, count)
    power = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * power
        if raw <= step:
            break
    start = math.floor(low / step) * step
    ticks = []
    value = start
    # Guarded rather than while-true: a step that underflows would spin, and
    # this runs in a request handler during a live demonstration.
    for _ in range(count * 4 + 4):
        if value > high + step * 0.5:
            break
        if value >= low - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


class Scale(object):
    """Map a data range onto a pixel range, with a padded, non-degenerate domain.

    The padding is why this is a class rather than a lambda: a series whose
    values are all equal has a zero-width domain, and dividing by it is the
    classic way a plot of a constant becomes a traceback.
    """

    def __init__(
        self,
        low: float,
        high: float,
        pixel_low: float,
        pixel_high: float,
        pad: float = 0.05,
    ) -> None:
        if not (high > low):
            # A constant series gets a unit window around its value, so the line
            # draws through the middle of the panel instead of on its edge.
            middle = high
            spread = abs(middle) * 0.1 or 1.0
            low, high = middle - spread, middle + spread
        else:
            margin = (high - low) * pad
            low, high = low - margin, high + margin
        self.low = low
        self.high = high
        self.pixel_low = pixel_low
        self.pixel_high = pixel_high

    def __call__(self, value: float) -> float:
        fraction = (value - self.low) / (self.high - self.low)
        return self.pixel_low + fraction * (self.pixel_high - self.pixel_low)


def _frame(
    title: str,
    subtitle: str,
    body: str,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> str:
    """Wrap plot body markup in a titled, labelled SVG element.

    ``role="img"`` with a ``<title>`` and ``<desc>`` is what makes the plot
    announce itself rather than read as an unlabelled graphic.
    """
    return (
        '<svg class="viz" viewBox="0 0 {w} {h}" width="100%" role="img" '
        'preserveAspectRatio="xMidYMid meet" aria-labelledby="">'
        "<title>{t}</title><desc>{d}</desc>{body}</svg>".format(
            w=width, h=height, t=escape(title), d=escape(subtitle), body=body
        )
    )


def empty_plot(title: str, message: str) -> str:
    """The panel a run with nothing to plot gets, in place of a plot.

    This is the degradation path, and it is a normal outcome rather than an
    error: a run whose config enabled no collector has no device series, and
    saying so is the correct rendering.
    """
    return _frame(
        title,
        message,
        '<text class="viz-empty" x="{x}" y="{y}" text-anchor="middle">{m}</text>'.format(
            x=PLOT_WIDTH // 2, y=PLOT_HEIGHT // 2, m=escape(message)
        ),
    )


def _axes(
    x_scale: Scale,
    y_scale: Scale,
    x_ticks: List[float],
    y_ticks: List[float],
    x_label: str,
    y_label: str,
    x_names: Optional[Dict[float, str]] = None,
) -> str:
    """Grid lines, tick labels and axis titles. Recessive by design."""
    plot_right = PLOT_WIDTH - MARGIN_RIGHT
    plot_bottom = PLOT_HEIGHT - MARGIN_BOTTOM
    parts = []
    for tick in y_ticks:
        y = y_scale(tick)
        if not (MARGIN_TOP - 1 <= y <= plot_bottom + 1):
            continue
        parts.append(
            '<line class="viz-grid" x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}"/>'.format(
                x1=MARGIN_LEFT, x2=plot_right, y=y
            )
        )
        parts.append(
            '<text class="viz-tick" x="{x}" y="{y:.1f}" text-anchor="end" '
            'dominant-baseline="middle">{t}</text>'.format(
                x=MARGIN_LEFT - 8, y=y, t=escape(fmt(tick))
            )
        )
    for tick in x_ticks:
        x = x_scale(tick)
        if not (MARGIN_LEFT - 1 <= x <= plot_right + 1):
            continue
        label = (x_names or {}).get(tick, fmt(tick))
        parts.append(
            '<text class="viz-tick" x="{x:.1f}" y="{y}" text-anchor="middle">{t}</text>'.format(
                x=x, y=plot_bottom + 18, t=escape(label)
            )
        )
    parts.append(
        '<line class="viz-axis" x1="{l}" y1="{b}" x2="{r}" y2="{b}"/>'.format(
            l=MARGIN_LEFT, r=plot_right, b=plot_bottom
        )
    )
    parts.append(
        '<text class="viz-axis-label" x="{x}" y="{y}" text-anchor="middle">{t}</text>'.format(
            x=(MARGIN_LEFT + plot_right) // 2, y=PLOT_HEIGHT - 12, t=escape(x_label)
        )
    )
    parts.append(
        '<text class="viz-axis-label" x="{x}" y="{y}" text-anchor="middle" '
        'transform="rotate(-90 {x} {y})">{t}</text>'.format(
            x=16, y=(MARGIN_TOP + plot_bottom) // 2, t=escape(y_label)
        )
    )
    return "".join(parts)


def _legend(names: Sequence[str], y: Optional[int] = None) -> str:
    """A swatch and a name per series, in the series' own colour order.

    Always emitted for two or more series: identity must not rest on colour
    alone, and the swatch is the only thing wearing the series colour - the name
    beside it stays in the page's ink.
    """
    if len(names) < 2:
        return ""
    parts = []
    x = MARGIN_LEFT
    top = MARGIN_TOP - 4 if y is None else y
    for index, name in enumerate(names):
        colour = SERIES_COLOURS[index % MAX_SERIES]
        parts.append(
            '<rect class="viz-swatch" x="{x}" y="{y}" width="10" height="10" rx="2" '
            'fill="{c}"/>'.format(x=x, y=top - 8, c=colour)
        )
        parts.append(
            '<text class="viz-legend" x="{x}" y="{y}" dominant-baseline="middle">'
            "{n}</text>".format(x=x + 15, y=top - 3, n=escape(name))
        )
        x += 24 + 7 * len(str(name))
    return "".join(parts)


def crossover_plot(series: List[Dict[str, Any]], x_label: str, y_label: str, title: str) -> str:
    """The metric against the swept axis, one line and marker set per placement.

    This is the sweep view's crossover: the point of the plot is where two
    placements' lines cross, so the markers are drawn over the lines with a
    surface ring and every marker carries a hover title naming its cell. Each
    series is also labelled at its last point when there are few enough for the
    labels not to collide, so the lines are identifiable without the legend.

    ``series`` is ``[{"name": str, "points": [(x, y, label), ...]}]``.
    """
    usable = [s for s in series if s.get("points")]
    if not usable:
        return empty_plot(title, "no cell in this sweep has both a metric and an axis value")
    if len(usable) > MAX_SERIES:
        # A ninth series is never a generated hue. Keeping the first eight and
        # saying how many were left out is honest; inventing a colour is not.
        hidden = len(usable) - MAX_SERIES
        usable = usable[:MAX_SERIES]
        title = "{} (first {} of {} placements)".format(title, MAX_SERIES, MAX_SERIES + hidden)

    xs = [p[0] for s in usable for p in s["points"]]
    ys = [p[1] for s in usable for p in s["points"]]
    plot_right = PLOT_WIDTH - MARGIN_RIGHT
    plot_bottom = PLOT_HEIGHT - MARGIN_BOTTOM
    x_scale = Scale(min(xs), max(xs), MARGIN_LEFT, plot_right)
    y_scale = Scale(min(ys), max(ys), plot_bottom, MARGIN_TOP)

    parts = [
        _axes(
            x_scale,
            y_scale,
            nice_ticks(x_scale.low, x_scale.high),
            nice_ticks(y_scale.low, y_scale.high),
            x_label,
            y_label,
        )
    ]
    for index, item in enumerate(usable):
        colour = SERIES_COLOURS[index % MAX_SERIES]
        points = item["points"]
        coords = [(x_scale(p[0]), y_scale(p[1])) for p in points]
        if len(coords) > 1:
            path = " ".join(
                "{}{:.1f},{:.1f}".format("M" if i == 0 else "L", x, y)
                for i, (x, y) in enumerate(coords)
            )
            parts.append(
                '<path class="viz-line" d="{d}" fill="none" stroke="{c}" '
                'stroke-width="{w}"/>'.format(d=path, c=colour, w=LINE_WIDTH)
            )
        for (x, y), point in zip(coords, points):
            label = point[2] if len(point) > 2 else ""
            parts.append(
                '<circle class="viz-marker" cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" '
                'stroke-width="{ring}"><title>{t}</title></circle>'.format(
                    x=x,
                    y=y,
                    r=MARKER_RADIUS,
                    c=colour,
                    ring=MARKER_RING,
                    t=escape(
                        "{}: {} = {}{}".format(
                            item["name"], label or fmt(point[0]), fmt(point[1]), ""
                        )
                    ),
                )
            )
        # Direct labels only while they stay legible; past four series the
        # legend carries identity on its own.
        if len(usable) <= 4 and coords:
            x, y = coords[-1]
            parts.append(
                '<text class="viz-direct" x="{x:.1f}" y="{y:.1f}" text-anchor="end" '
                'dominant-baseline="auto">{n}</text>'.format(
                    x=min(x, plot_right), y=max(MARGIN_TOP + 10, y - 9), n=escape(item["name"])
                )
            )
    parts.append(_legend([str(s["name"]) for s in usable]))
    return _frame(title, "{} against {}".format(y_label, x_label), "".join(parts))


def bar_panel(
    values: List[Tuple[str, Optional[float]]],
    title: str,
    value_label: str,
    height: Optional[int] = None,
) -> str:
    """A horizontal bar per named value, anchored to a common baseline.

    Horizontal because the names are words (a stage name, a node name) and a
    rotated tick label is harder to read than a longer panel. A value that is
    absent draws no bar and prints a dash, so a missing measurement is visibly
    missing rather than a zero-length bar indistinguishable from a real zero.
    """
    present = [(name, value) for name, value in values if value is not None]
    if not present:
        return empty_plot(title, "nothing measured for {}".format(value_label))
    row_height = 34
    chart_height = height or (MARGIN_TOP + 24 + row_height * len(values) + 28)
    plot_right = PLOT_WIDTH - MARGIN_RIGHT - 60
    top = MARGIN_TOP + 16
    scale = Scale(0.0, max(v for _n, v in present), MARGIN_LEFT, plot_right, pad=0.0)

    parts = []
    for tick in nice_ticks(0.0, scale.high, 4):
        x = scale(tick)
        parts.append(
            '<line class="viz-grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y2}"/>'.format(
                x=x, y1=top - 8, y2=top + row_height * len(values) - 10
            )
        )
        parts.append(
            '<text class="viz-tick" x="{x:.1f}" y="{y}" text-anchor="middle">{t}</text>'.format(
                x=x, y=top + row_height * len(values) + 6, t=escape(fmt(tick))
            )
        )
    for index, (name, value) in enumerate(values):
        y = top + index * row_height
        parts.append(
            '<text class="viz-tick" x="{x}" y="{y}" text-anchor="end" '
            'dominant-baseline="middle">{n}</text>'.format(
                x=MARGIN_LEFT - 8, y=y + 8, n=escape(name)
            )
        )
        if value is None:
            parts.append(
                '<text class="viz-empty" x="{x}" y="{y}" dominant-baseline="middle">'
                "not measured</text>".format(x=MARGIN_LEFT + 4, y=y + 8)
            )
            continue
        width = max(2.0, scale(value) - MARGIN_LEFT)
        parts.append(
            '<rect class="viz-bar" x="{x}" y="{y}" width="{w:.1f}" height="16" rx="{r}" '
            'fill="{c}"><title>{t}</title></rect>'.format(
                x=MARGIN_LEFT,
                y=y,
                w=width,
                r=BAR_RADIUS,
                c=SERIES_COLOURS[0],
                t=escape("{}: {} {}".format(name, fmt(value), value_label)),
            )
        )
        parts.append(
            '<text class="viz-direct" x="{x:.1f}" y="{y}" dominant-baseline="middle">'
            "{v}</text>".format(x=MARGIN_LEFT + width + 6, y=y + 8, v=escape(fmt(value)))
        )
    parts.append(
        '<text class="viz-axis-label" x="{x}" y="{y}" text-anchor="middle">{t}</text>'.format(
            x=(MARGIN_LEFT + plot_right) // 2, y=chart_height - 6, t=escape(value_label)
        )
    )
    return _frame(title, value_label, "".join(parts), height=chart_height)


def series_panel(
    series: List[Dict[str, Any]],
    title: str,
    y_label: str,
    x_label: str = "seconds into the run",
) -> str:
    """One or more quantities over time, sharing one y axis.

    One axis, always: two quantities of different scale get two panels rather
    than a second y scale, so a caller groups by unit before calling here.

    ``series`` is ``[{"name": str, "t": [...], "y": [...]}]``.
    """
    usable = [s for s in series if s.get("y")]
    if not usable:
        return empty_plot(title, "no device samples in this run")
    if len(usable) > MAX_SERIES:
        usable = usable[:MAX_SERIES]
    ts = [t for s in usable for t in s["t"]]
    ys = [y for s in usable for y in s["y"]]
    plot_right = PLOT_WIDTH - MARGIN_RIGHT
    plot_bottom = PLOT_HEIGHT - MARGIN_BOTTOM
    x_scale = Scale(min(ts), max(ts), MARGIN_LEFT, plot_right, pad=0.0)
    y_scale = Scale(min(ys), max(ys), plot_bottom, MARGIN_TOP)

    parts = [
        _axes(
            x_scale,
            y_scale,
            nice_ticks(x_scale.low, x_scale.high),
            nice_ticks(y_scale.low, y_scale.high),
            x_label,
            y_label,
        )
    ]
    for index, item in enumerate(usable):
        colour = SERIES_COLOURS[index % MAX_SERIES]
        pairs = list(zip(item["t"], item["y"]))
        # A collector at 10 ms writes tens of thousands of samples; one path
        # element per sample would be megabytes of markup for a line no wider
        # than the panel, so the series is thinned to about two points per pixel.
        budget = int((plot_right - MARGIN_LEFT) * 2)
        if len(pairs) > budget:
            stride = len(pairs) // budget + 1
            pairs = pairs[::stride] + [pairs[-1]]
        path = " ".join(
            "{}{:.1f},{:.1f}".format("M" if i == 0 else "L", x_scale(t), y_scale(y))
            for i, (t, y) in enumerate(pairs)
        )
        parts.append(
            '<path class="viz-line" d="{d}" fill="none" stroke="{c}" stroke-width="{w}">'
            "<title>{n}</title></path>".format(
                d=path, c=colour, w=LINE_WIDTH, n=escape(item["name"])
            )
        )
    parts.append(_legend([str(s["name"]) for s in usable]))
    return _frame(title, y_label, "".join(parts))

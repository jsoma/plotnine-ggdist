"""
Test helpers: image comparison and layer data

Image tests compare a plot with a baseline image using
``assert p == "name"``, as in plotnine's own test suite and in
plotnine-ggtext. Results are written to ``tests/result_images/<module>/``
and compared with ``tests/baseline_images/<module>/<name>.png``.

Create or update baselines with ``pytest --update-baselines``.
"""

from __future__ import annotations

import inspect
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
from plotnine import ggplot, theme_set
from plotnine.options import set_option
from plotnine.themes.theme import DEFAULT_RCPARAMS

TOLERANCE = 2  # plotnine's default
DPI = 72

# fonts that ship with matplotlib, for identical images everywhere
DEFAULT_RCPARAMS["font.monospace"] = ["DejaVu Sans Mono"]
DEFAULT_RCPARAMS["font.sans-serif"] = ["DejaVu Sans"]
DEFAULT_RCPARAMS["font.serif"] = ["DejaVu Serif"]

# 640 x 480 px, the size of the R renderings in tests/reference_r/
set_option("dpi", DPI)
set_option("figure_size", (640 / DPI, 480 / DPI))

TESTS_DIR = Path(__file__).parent
BASELINE_DIR = TESTS_DIR / "baseline_images"
RESULT_DIR = TESTS_DIR / "result_images"
DATA_DIR = TESTS_DIR / "data"

_UPDATE = False


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Write result images as the new baselines",
    )


def pytest_configure(config):
    global _UPDATE
    _UPDATE = config.getoption("--update-baselines")


def _ggplot_eq(plot: ggplot, name: str) -> bool:
    module = Path(inspect.stack()[1].filename).stem
    png = Path(name).with_suffix(".png").name
    result = RESULT_DIR / module / png
    baseline = BASELINE_DIR / module / png
    expected = RESULT_DIR / module / f"{Path(png).stem}-expected.png"
    result.parent.mkdir(parents=True, exist_ok=True)

    plt.close("all")
    mpl.use("Agg")
    with mpl.rc_context({"text.hinting": "auto", "text.antialiased": True}):
        plot.save(result, verbose=False)
    plt.close("all")
    warnings.resetwarnings()

    if _UPDATE:
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result, baseline)
    if not baseline.exists():
        raise FileNotFoundError(
            f"Baseline image {baseline} is missing; run pytest --update-baselines"
        )
    shutil.copyfile(baseline, expected)
    err = compare_images(str(expected), str(result), TOLERANCE, in_decorator=True)
    plot._err = err  # pyright: ignore[reportAttributeAccessIssue]
    return not err


ggplot.__eq__ = _ggplot_eq  # pyright: ignore[reportAttributeAccessIssue]


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, ggplot) and isinstance(right, str) and op == "==":
        err = getattr(left, "_err", None)
        if err:
            return ["images not close: {actual} vs. {expected} (RMS {rms:.2f})".format(**err)]


def _layer_data(plot: ggplot, i: int = 0) -> pd.DataFrame:
    p = deepcopy(plot)
    p._build()
    return p.layers[i].data


@pytest.fixture
def layer_data():
    """
    ``layer_data(plot, i=0)``: the data of layer `i` after the plot is built
    (like ggplot2's `layer_data()`)
    """
    return _layer_data


@pytest.fixture(autouse=True)
def _reset():
    from plotnine import theme_gray

    theme_set(theme_gray())
    yield
    plt.close("all")

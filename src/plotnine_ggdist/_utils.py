"""
Internal helpers shared by the stats
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

DIST_COLUMNS = ("xdist", "ydist")


def _uniquecols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns with a single value, as a one-row frame

    plotnine's own version sorts values, which fails for columns of
    distribution objects; this one hashes them instead.
    """
    keep = []
    for c in df.columns:
        if c in DIST_COLUMNS:
            continue
        try:
            n = df[c].nunique(dropna=False)
        except TypeError:  # unhashable values
            continue
        if n == 1:
            keep.append(c)
    return df[keep].iloc[[0]].reset_index(drop=True)


def compute_panel_by_group(
    data: pd.DataFrame, scales: Any, compute_group: Callable[[pd.DataFrame, Any], pd.DataFrame]
) -> pd.DataFrame:
    """
    plotnine's stat.compute_panel, safe for distribution columns
    """
    if not len(data):
        return type(data)()
    out = []
    for _, old in data.groupby("group"):
        new = compute_group(old, scales).reset_index(drop=True)
        if not len(new):
            continue
        unique = _uniquecols(old)
        missing = unique.columns.difference(new.columns)
        u = unique.loc[[0] * len(new), missing].reset_index(drop=True)
        out.append(pd.concat([new, u], axis=1))
    if not out:
        return type(data)()
    return pd.concat(out, axis=0, ignore_index=True)


def _is_discrete(scale: Any) -> bool:
    from plotnine.scales.scale_discrete import scale_discrete

    return isinstance(scale, scale_discrete)


def width_aware_layer(lyr: Any) -> Any:
    """
    Give a layer of one of our geoms ggdist's conveniences

    - plotnine keeps only mapped columns, but ggdist's geoms use ``.width``
      (e.g. from median_qi()) without it being mapped: to size intervals
      and to colour ribbons. If the layer data has ``.width`` and it is not
      mapped, map it to itself.
    - If the geom colours by ``.width`` (its stat has a ``width_aes``
      parameter) and that aesthetic is not mapped, map it to
      ``after_stat("level")`` and label it "level".
    - Record which position scale is discrete, so the geom can detect its
      orientation from it (``geom._scale_orientation``).
    """
    from plotnine.layer import layer
    from plotnine.mapping.aes import aes, make_labels
    from plotnine.mapping.evaluation import after_stat

    class _ggdist_layer(type(lyr)):  # type: ignore[misc]
        def setup(self, plot):
            super().setup(plot)
            data = self.data
            if not (isinstance(data, pd.DataFrame) and ".width" in data.columns):
                return
            if ".width" not in self.mapping:
                self.mapping[".width"] = ".width"
            target = self.stat.params.get("width_aes")
            if target and target not in self.mapping and target not in self.geom.aes_params:
                self.mapping[target] = after_stat("level")
                plot.labels.add_defaults(make_labels(aes(**{target: after_stat("level")})))

        def compute_statistic(self, layout):
            sx = layout.panel_scales_x[0] if layout.panel_scales_x else None
            sy = layout.panel_scales_y[0] if layout.panel_scales_y else None
            hint = None
            if _is_discrete(sy) and not _is_discrete(sx):
                hint = "horizontal"
            elif _is_discrete(sx) and not _is_discrete(sy):
                hint = "vertical"
            self.geom._scale_orientation = hint
            super().compute_statistic(layout)

    assert isinstance(lyr, layer)
    lyr.__class__ = _ggdist_layer
    return lyr

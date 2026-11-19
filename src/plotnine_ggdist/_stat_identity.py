"""
An identity stat that maps ``.width`` to colour by default

ggdist's geom_lineribbon() and geom_interval() fill / colour their
ribbons and intervals by ``.width`` when the data has that column (e.g.
the output of median_qi()). This stat reproduces that: it adds an ordered
``level`` column (widest first) and, unless the user mapped the target
aesthetic, maps it to ``after_stat("level")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plotnine.stats.stat_identity import stat_identity

from .stat_slabinterval import as_level, width_levels

if TYPE_CHECKING:
    import pandas as pd
    from plotnine.facets.layout import Layout


class stat_width_identity(stat_identity):
    """
    Identity stat; maps ``level`` (from ``.width``) to `width_aes`
    """

    DEFAULT_PARAMS = stat_identity.DEFAULT_PARAMS | {"width_aes": "fill"}

    def compute_layer(self, data: pd.DataFrame, layout: Layout) -> pd.DataFrame:
        # the layer maps `width_aes` to after_stat("level") when the data
        # has .width and the aesthetic is not mapped
        if ".width" in data.columns and "level" not in data.columns:
            data = data.copy()
            data["level"] = as_level(data[".width"], width_levels(data[".width"].dropna()))
        return data


class stat_width_identity_color(stat_width_identity):
    """
    Identity stat; maps ``level`` (from ``.width``) to colour
    """

    DEFAULT_PARAMS = stat_width_identity.DEFAULT_PARAMS | {"width_aes": "color"}

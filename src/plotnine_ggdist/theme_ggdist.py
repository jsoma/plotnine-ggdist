"""
theme_ggdist: a minimal theme suited to distribution plots
"""

from __future__ import annotations

from plotnine import element_blank, element_line, element_rect, element_text, theme, theme_light

__all__ = ("theme_ggdist",)


class theme_ggdist(theme_light):
    """
    A simple theme with light axis lines and no grid on the group axis

    Modelled on `ggdist::theme_ggdist()`: a light theme without panel
    border or minor grid lines, axis lines in light grey, and facet strips
    as plain dark text.

    Parameters
    ----------
    base_size :
        Base font size, points.
    base_family :
        Base font family.
    """

    def __init__(self, base_size: float = 11, base_family: str | None = None):
        super().__init__(base_size, base_family)
        self += theme(
            panel_border=element_blank(),
            panel_grid=element_blank(),
            legend_key=element_blank(),
            axis_line=element_line(color="#bdbdbd", size=0.5),
            axis_ticks=element_line(color="#bdbdbd", size=0.5),
            strip_background=element_rect(fill="none", color="none"),
            strip_text=element_text(color="#262626", ha="left"),
        )

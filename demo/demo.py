"""
The plot in the README: python demo/demo.py
"""

import numpy as np
import pandas as pd
from plotnine import aes, after_stat, geom_vline, ggplot, labs, scale_fill_brewer, scale_fill_manual, theme
from scipy import stats

from plotnine_ggdist import stat_halfeye, stat_lineribbon, theme_ggdist

rng = np.random.default_rng(2026)

# posterior draws of three treatment effects
effects = pd.DataFrame(
    {
        "treatment": np.repeat(["A", "B", "C"], 4000),
        "effect": np.r_[rng.normal(0.4, 0.5, 4000), rng.normal(1.1, 0.4, 4000), rng.gamma(4, 0.25, 4000) - 0.2],
    }
)
left = (
    ggplot(effects, aes(x="effect", y="treatment"))
    + geom_vline(xintercept=0, color="#999999", linetype="dashed")
    + stat_halfeye(aes(fill=after_stat("x > 0")), point_interval="median_hdi")
    + scale_fill_manual(values=["#D1495B", "#8FB8DE"], name="effect > 0")
    + labs(x="Treatment effect", y="", title="Posterior draws, median + HDI")
    + theme_ggdist()
)

# a forecast with growing uncertainty, from analytical distributions
months = pd.DataFrame({"month": np.arange(1, 13)})
months["forecast"] = [stats.t(5, loc=10 + 0.6 * m, scale=0.5 + 0.15 * m) for m in months["month"]]
right = (
    ggplot(months, aes(x="month", ydist="forecast"))
    + stat_lineribbon(width=[0.5, 0.8, 0.95])
    + scale_fill_brewer(palette="Blues")
    + labs(x="Month", y="Sales", title="Forecast from scipy.stats.t(...)")
    + theme_ggdist()
)

p = left | right
p = p & theme(figure_size=(11, 4.2), dpi=110)
p.save("demo.png", verbose=False)
print("saved demo.png")

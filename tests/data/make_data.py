"""
Generate the sample data shared by the Python tests and the R reference
script (tests/reference_r/render.R), so both render the same draws.

Mirrors the data in ggdist's roxygen examples; numpy cannot reproduce R's
random numbers, so the draws are saved instead.

    python tests/data/make_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(1234)

# df = data.frame(group = c("a", "b", "c"),
#                 value = rnorm(1500, mean = c(5, 7, 9), sd = c(1, 1.5, 1)))
pd.DataFrame(
    {
        "group": np.tile(["a", "b", "c"], 500),
        "value": rng.normal(np.tile([5, 7, 9], 500), np.tile([1, 1.5, 1], 500)).round(6),
    }
).to_csv(HERE / "samples.csv", index=False)

# stat_slabinterval's examples:
# data.frame(group = c("a", "b", "c", "c", "c"),
#            value = rnorm(2500, mean = c(5, 7, 9, 9, 9), sd = c(1, 1.5, 1, 1, 1)))
pd.DataFrame(
    {
        "group": np.tile(["a", "b", "c", "c", "c"], 500),
        "value": rng.normal(np.tile([5, 7, 9, 9, 9], 500), np.tile([1, 1.5, 1, 1, 1], 500)).round(6),
    }
).to_csv(HERE / "samples5.csv", index=False)

# data.frame(group = letters[1:3], value = rnorm(3000, 3:1))
pd.DataFrame(
    {
        "group": np.tile(["a", "b", "c"], 1000),
        "value": rng.normal(np.tile([3, 2, 1], 1000)).round(6),
    }
).to_csv(HERE / "ridge.csv", index=False)

# tibble(x = rep(1:10, 100), y = rnorm(1000, x))
x = np.tile(np.arange(1, 11), 100)
pd.DataFrame({"x": x, "y": rng.normal(x).round(6)}).to_csv(HERE / "line.csv", index=False)

# stand-in for ggdist's RankCorr_u_tau data (not redistributed here)
i = np.repeat(np.arange(1, 6), 400)
pd.DataFrame({"i": i, "u_tau": rng.normal(0.3 * i, 0.4 + 0.05 * i).round(6)}).to_csv(
    HERE / "ranks.csv", index=False
)

# a bimodal sample
pd.DataFrame(
    {"x": np.r_[rng.normal(0, 1, 2000), rng.normal(5, 1, 2000)].round(6)}
).to_csv(HERE / "bimodal.csv", index=False)
print("wrote", sorted(p.name for p in HERE.glob("*.csv")))

# dots examples (appended so the draws above are unchanged):
# tibble(g = rep(c("a", "b"), 200), value = rnorm(400, c(0, 3), c(0.75, 1)))
pd.DataFrame(
    {
        "g": np.tile(["a", "b"], 200),
        "value": rng.normal(np.tile([0, 3], 200), np.tile([0.75, 1], 200)).round(6),
    }
).to_csv(HERE / "dots.csv", index=False)
print("wrote dots.csv")

# smoothing, bar and blur examples (appended so the draws above are unchanged)
pd.DataFrame({"x": rng.normal(size=1000).round(6)}).to_csv(HERE / "normal1000.csv", index=False)
pd.DataFrame({"x": rng.poisson(2, 1000)}).to_csv(HERE / "poisson.csv", index=False)
pd.DataFrame({"x_beta": rng.beta(0.5, 0.5, 1000).round(6)}).to_csv(HERE / "beta.csv", index=False)
# the vignette's abcd_df: sample(c("a","b","c","d"), 1000, replace = TRUE,
#   prob = c(0.27, 0.6, 0.03, 0.005)), g = rep(c("a", "b"), 500)
pd.DataFrame(
    {
        "x": rng.choice(["a", "b", "c", "d"], 1000, p=np.array([0.27, 0.6, 0.03, 0.005]) / 0.905),
        "g": np.tile(["a", "b"], 500),
    }
).to_csv(HERE / "abcd.csv", index=False)
print("wrote normal1000.csv, poisson.csv, beta.csv, abcd.csv")

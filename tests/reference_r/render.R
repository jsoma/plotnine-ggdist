# Render ggdist's versions of the image tests in tests/test_examples.py
#
#   Rscript tests/reference_r/render.R    (from the repository root)
#
# Needs ggplot2, ggdist, distributional, dplyr, posterior and ragg. Uses the same
# draws as the Python tests (tests/data/*.csv) so the two can be compared
# side by side. Images are 640 x 480 px at 72 dpi, like the baselines.
#
# These are reference renderings only; the R images are not test inputs.

library(ggplot2)
library(ggdist)
library(distributional)
library(dplyr, warn.conflicts = FALSE)

outdir <- "tests/reference_r"
datadir <- "tests/data"
save_png <- function(name, plot) {
  ggsave(file.path(outdir, paste0(name, ".png")), plot,
         width = 640 / 72, height = 480 / 72, dpi = 72, device = ragg::agg_png)
}

theme_set(theme_ggdist())

df <- read.csv(file.path(datadir, "samples.csv"))
ridge <- read.csv(file.path(datadir, "ridge.csv"))
line <- read.csv(file.path(datadir, "line.csv"))
ranks <- read.csv(file.path(datadir, "ranks.csv"))
bimodal <- read.csv(file.path(datadir, "bimodal.csv"))

dist_df <- data.frame(group = c("a", "b", "c"), mean = c(5, 7, 8), sd = c(1, 1.5, 1))
line_dist <- data.frame(x = 1:10, sd = seq(1, 3, length.out = 10))
grid <- expand.grid(mean = 1:3, input = seq(-2, 6, length.out = 100)) |>
  mutate(group = letters[4 - mean], density = dnorm(input, mean, 1))

samples_plot <- ggplot(df, aes(x = value, y = group))
dist_plot <- ggplot(dist_df, aes(y = group, xdist = dist_normal(mean, sd)))

# ports of the roxygen examples ------------------------------------------------
shortcuts <- list(
  stat_halfeye = stat_halfeye, stat_cdfinterval = stat_cdfinterval, stat_eye = stat_eye, stat_slab = stat_slab,
  stat_pointinterval = stat_pointinterval,
  stat_gradientinterval = stat_gradientinterval,
  stat_histinterval = stat_histinterval
)
for (name in names(shortcuts)) {
  save_png(paste0(name, "-1"), samples_plot + shortcuts[[name]]())
  if (name != "stat_histinterval") {
    save_png(paste0(name, "-2"), dist_plot + shortcuts[[name]]())
  }
}

samples5 <- read.csv(file.path(datadir, "samples5.csv"))
save_png("stat_slabinterval-1", ggplot(samples5, aes(x = group, y = value)) + stat_eye())
save_png("stat_slabinterval-2",
  ggplot(samples5, aes(x = group, y = value)) + stat_eye(aes(thickness = after_stat(pdf * n))))
dist_df5 <- tibble::tribble(
  ~group, ~subgroup, ~mean, ~sd,
  "a", "h", 5, 1,
  "b", "h", 7, 1.5,
  "c", "h", 8, 1,
  "c", "i", 9, 1,
  "c", "j", 7, 1
)
save_png("stat_slabinterval-3",
  ggplot(dist_df5, aes(x = group, ydist = dist_normal(mean, sd), fill = subgroup)) +
    stat_eye(position = "dodge"))
save_png("stat_slabinterval-5",
  data.frame(dist = "lnorm", logmean = log(10), logsd = 2 * log(10)) |>
    ggplot(aes(y = 1, dist = dist, arg1 = logmean, arg2 = logsd)) +
    stat_halfeye() +
    scale_x_log10(breaks = 10^seq(-5, 7, by = 2)))

save_png("stat_slab-3",
  ggplot(ridge, aes(y = group, x = value)) +
    stat_slab(color = "black", expand = TRUE, trim = FALSE, height = 2))

save_png("stat_interval-1", samples_plot + stat_interval() + scale_color_brewer())
save_png("stat_interval-2", dist_plot + stat_interval() + scale_color_brewer())
save_png("stat_ccdfinterval-1", samples_plot + stat_ccdfinterval() + expand_limits(x = 0))
save_png("stat_ccdfinterval-2", dist_plot + stat_ccdfinterval() + expand_limits(x = 0))

save_png("stat_lineribbon-1",
  ggplot(line, aes(x = x, y = y)) + stat_lineribbon() + scale_fill_brewer())
save_png("stat_lineribbon-2",
  ggplot(line_dist, aes(x = x, ydist = dist_normal(x, sd))) + stat_lineribbon() + scale_fill_brewer())
save_png("stat_ribbon-1",
  ggplot(line, aes(x = x, y = y)) + stat_ribbon() + scale_fill_brewer())
save_png("stat_ribbon-2",
  ggplot(line_dist, aes(x = x, ydist = dist_normal(x, sd))) + stat_ribbon() + scale_fill_brewer())

save_png("geom_lineribbon-1",
  line |> group_by(x) |> median_qi(.width = c(.5, .8, .95)) |>
    ggplot(aes(x = x, y = y, ymin = .lower, ymax = .upper)) +
    geom_lineribbon() + scale_fill_brewer())

ranks_80_95 <- ranks |> group_by(i) |> median_qi(.width = c(.8, .95))
save_png("geom_pointinterval-1",
  ggplot(ranks_80_95, aes(y = i, x = u_tau, xmin = .lower, xmax = .upper)) + geom_pointinterval())
save_png("geom_pointinterval-2",
  ggplot(ranks_80_95, aes(x = i, y = u_tau, ymin = .lower, ymax = .upper)) + geom_pointinterval())

ranks_4 <- ranks |> group_by(i) |> median_qi(.width = c(.5, .8, .95, .99))
save_png("geom_interval-1",
  ggplot(ranks_4, aes(y = i, x = u_tau, xmin = .lower, xmax = .upper)) +
    geom_interval() + scale_color_brewer())
save_png("geom_interval-2",
  ggplot(ranks_4, aes(x = i, y = u_tau, ymin = .lower, ymax = .upper)) +
    geom_interval() + scale_color_brewer())

save_png("geom_slab-1", ggplot(grid, aes(y = group, x = input, thickness = density)) + geom_slab())
save_png("geom_slab-2", ggplot(grid, aes(x = group, y = input, thickness = density)) + geom_slab())
save_png("geom_slab-3",
  ggplot(grid, aes(y = group, x = input, thickness = density)) + geom_slab(height = 2, color = "black"))

# dot plots ---------------------------------------------------------------------
dots <- read.csv(file.path(datadir, "dots.csv"))
dot_geoms <- list(geom_dots = geom_dots, geom_dotsinterval = geom_dotsinterval,
                  geom_swarm = geom_swarm, geom_weave = geom_weave)
for (name in names(dot_geoms)) {
  save_png(paste0(name, "-1"), ggplot(dots, aes(x = value, y = g)) + dot_geoms[[name]]())
  save_png(paste0(name, "-2"), ggplot(dots, aes(y = value, x = g)) + dot_geoms[[name]]())
}
save_png("geom_dotsinterval-3",
  ggplot(ranks, aes(x = u_tau, y = factor(i))) + stat_dots(quantiles = 100))
save_png("geom_dotsinterval-4",
  ggplot(ranks, aes(x = u_tau, y = factor(i), fill = after_stat(x > 1))) +
    stat_dotsinterval(quantiles = 100))
save_png("stat_dots-1", ggplot(line, aes(x = x, y = y)) + stat_dots())
save_png("stat_dots-2",
  ggplot(line_dist, aes(x = x, ydist = dist_normal(x, sd))) + stat_dots(quantiles = 50))
save_png("stat_dotsinterval-1", ggplot(line, aes(x = x, y = y)) + stat_dotsinterval())
save_png("stat_dotsinterval-2",
  ggplot(line_dist, aes(x = x, ydist = dist_normal(x, sd))) + stat_dotsinterval(quantiles = 50))

save_png("raincloud",
  ggplot(head(df, 180), aes(x = group, y = value, fill = group)) +
    stat_slab(adjust = 0.5, height = 0.6, justification = -0.2) +
    geom_boxplot(width = 0.12, outlier.shape = NA) +
    stat_dots(side = "left", justification = 1.1, binwidth = 0.25))
save_png("quantile_dotplot_by_level",
  dist_plot + stat_dotsinterval(aes(fill = after_stat(level)), quantiles = 50, slab_color = "white") +
    scale_fill_brewer(na.value = "#cccccc"))
for (layout in c("bin", "weave", "hex", "swarm")) {
  save_png(paste0("layout_", layout),
    ggplot(dots, aes(x = value, y = g)) +
      geom_dots(layout = layout, side = "both", stackratio = if (layout == "hex") 0.9 else 1))
}
save_png("fixed_binwidth_compressed",
  ggplot(dots, aes(x = value, y = g)) + geom_dots(binwidth = unit(3, "mm"), overflow = "compress"))
save_png("discrete_quantile_dotplot",
  ggplot(data.frame(g = "poisson(4)"), aes(y = g, xdist = dist_poisson(4))) +
    stat_dotsinterval(quantiles = 40))

# smoothing, bar layout, blur ------------------------------------------------------
normal1000 <- read.csv(file.path(datadir, "normal1000.csv"))
poisson <- read.csv(file.path(datadir, "poisson.csv"))
beta <- read.csv(file.path(datadir, "beta.csv"))
abcd <- read.csv(file.path(datadir, "abcd.csv"))

save_png("smooth_density-1", ggplot(normal1000, aes(x)) + geom_dots())
save_png("smooth_density-2", ggplot(normal1000, aes(x)) + geom_dots(smooth = "unbounded"))
save_png("smooth_density-3",
  ggplot(normal1000, aes(x)) + geom_dots(smooth = smooth_unbounded(kernel = "triangular", adjust = 0.5)))
save_png("smooth_density-4",
  ggplot(beta, aes(x_beta)) + geom_dots(smooth = smooth_bounded(bounds = c(0, 1))))

save_png("smooth_discrete-1", ggplot(poisson, aes(x)) + geom_dots())
save_png("smooth_discrete-2", ggplot(poisson, aes(x)) + geom_dots(layout = "bar"))
save_png("smooth_discrete-3", ggplot(poisson, aes(x)) + geom_dots(smooth = "discrete"))
save_png("smooth_discrete-4", ggplot(poisson, aes(x)) + geom_dots(smooth = "bar"))
save_png("smooth_discrete-5",
  ggplot(poisson, aes(x)) +
    geom_dots(smooth = smooth_discrete(kernel = "epanechnikov", width = 0.8), side = "both"))

p_blur <- ppoints(100)
blur_df <- data.frame(q = quantile(normal1000$x, p_blur),
                      se = posterior::mcse_quantile(normal1000$x, p_blur))
save_png("geom_blur_dots-1", ggplot(blur_df, aes(x = q, sd = se)) + geom_blur_dots())
save_png("geom_blur_dots-2", ggplot(blur_df, aes(x = q, sd = se)) + geom_blur_dots(blur = "interval"))
save_png("stat_mcse_dots-1",
  ggplot(normal1000, aes(x = x)) + stat_mcse_dots(quantiles = 100, layout = "weave"))

save_png("bar_layout_stacked_fill",
  ggplot(abcd, aes(x = x, fill = g, order = g)) +
    geom_dots(layout = "bar", group = NA, color = NA) +
    scale_y_continuous(breaks = NULL))
save_png("pch_shapes",
  ggplot(data.frame(g = c("pch 1", "pch 16", "pch 21", "pch 22"), s = c(1, 16, 21, 22)),
         aes(y = g, xdist = dist_normal(0, 1), slab_shape = s)) +
    stat_dotsinterval(quantiles = 30, fill = "#8FB8DE", slab_color = "#1f4e79", shape = 21) +
    scale_shape_identity(aesthetics = "slab_shape"))

# beyond the examples -----------------------------------------------------------
save_png("fill_by_level",
  dist_plot + stat_halfeye(aes(fill = after_stat(level)), slab_color = "gray") +
    scale_fill_brewer(na.value = "#eeeeee"))
save_png("fill_by_expression", samples_plot + stat_halfeye(aes(fill = after_stat(x > 7))))
save_png("multimodal_hdi", ggplot(bimodal, aes(x = x)) + stat_halfeye(point_interval = "mode_hdi"))
save_png("log_scale_distributions",
  ggplot(data.frame(g = c("sdlog 0.5", "sdlog 1"), s = c(0.5, 1)),
         aes(y = g, xdist = dist_lognormal(0, s))) +
    stat_halfeye() + scale_x_log10())
save_png("discrete_distributions",
  ggplot(data.frame(g = c("poisson(3)", "binom(10, 0.3)")),
         aes(y = g, xdist = c(dist_poisson(3), dist_binomial(10, 0.3)))) +
    stat_halfeye())
save_png("facets_and_coord_flip",
  df |> mutate(half = ifelse(row_number() <= n() / 2, "first", "second")) |>
    ggplot(aes(x = value, y = group)) + stat_halfeye() + facet_wrap(~half) + coord_flip())
save_png("side_bottom_and_subgeometry_colors",
  samples_plot + stat_halfeye(aes(fill = group), side = "bottom",
                              interval_color = "gray", point_color = "black"))
save_png("dodged_pointintervals",
  df |> mutate(cond = rep(c("ctrl", "treat"), length.out = n())) |>
    ggplot(aes(x = group, y = value, color = cond)) +
    stat_pointinterval(position = position_dodge(width = 0.4)))
save_png("gradient_segments", dist_plot + stat_gradientinterval(fill_type = "segments", fill = "#2166ac"))

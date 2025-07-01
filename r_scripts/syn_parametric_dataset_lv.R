library(sjPlot)
library(broom.mixed)    # for tidy()
library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
library(moments)
library(ggeffects)
library(robustlmm)
library(glmmTMB)
library(patchwork)  
library(forcats)
library(scales)


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_plot_long_df_with_lv.csv", 
               stringsAsFactors = TRUE)
#summary(df)
#head(df)

# relevel
df$method <- relevel(df$method, ref = "acf_full")
# scale
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    lv_s         = as.numeric(scale(lv)),
    tau_ms_true_s   = as.numeric(scale(tau_ms_true))
  )

# log‐transform, check tails - kurtosis still high
df <- df %>%
  mutate(
    log_tau_diff = log10(tau_diff_rel)
  )

# skew_log  <- skewness(df$log_tau_diff, na.rm = TRUE)
# kurt_log  <- kurtosis(df$log_tau_diff, na.rm = TRUE)
# cat("skewness:", round(skew_log, 3), "\n")   # should be closer to 0
# cat("kurtosis:", round(kurt_log, 3), "\n")   # should be nearer 3
# 
# ggplot(df, aes(x = log_tau_diff)) +
#   geom_histogram(aes(y = ..density..), bins = 30, fill = "grey80", color = "black") +
#   geom_density() +
#   labs(title = "Histogram & Density of log(tau_diff_rel)",
#        x     = "log_tau_diff",
#        y     = "Density")
# 
# ggplot(df, aes(sample = log_tau_diff)) +
#   stat_qq() +
#   stat_qq_line() +
#   labs(title = "QQ‐plot log_tau_diff")

# fit model

model_log <- lmer(
  log_tau_diff ~ method * (fr_s + lv_s + tau_ms_true_s)
  + (1 | unit_id),
  data = df,
  REML = FALSE
)
# check residuals - heavy tail present!
res  <- residuals(model_log)

ggplot(data.frame(res), aes(x = res)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res); qqline(res)

res_skew <- skewness(res)
res_kurt <- kurtosis(res)
cat("Residual skewness:", round(res_skew, 3), "\n")
cat("Residual kurtosis:", round(res_kurt, 3), "\n")


# FAILED: Fit a robust mixed‐effects mode lfrom robustlmm
# fails with cannot allocate vector of size 74.0 Gb
# model_rob <- rlmer(
#   log_tau_diff ~ method * (fr_s + alpha_s + tau_ms_true_s)
#   + (1 | unit_id),
#   data = df
# )


# FAILED: Fit model with Student-t distribution for residuals 
# cannot converge
# model_t <- glmmTMB(
#   log_tau_diff ~ method * (fr_s + alpha_s + tau_ms_true_s)
#   + (1 | unit_id),
#   data   = df,
#   family = t_family(link = "identity")      # Student‐t residuals
# )

# Winsorizing

# 2.5% and 97.5% cutoffs
qs <- quantile(df$log_tau_diff, probs = c(0.025, 0.975), na.rm = TRUE)
lo <- qs[1]; hi <- qs[2]
# pull everything below lo up to lo, above hi down to hi
df <- df %>%
  mutate(
    log_tau_win = pmax(pmin(log_tau_diff, hi), lo)
  )

model_win <- lmer(
  log_tau_win ~ method * (fr_s + lv_s + tau_ms_true_s)
  + (1 | unit_id),
  data   = df,
  REML   = FALSE
)

summary(model_win)

# check residuals 
res_win  <- residuals(model_win)

ggplot(data.frame(res_win), aes(x = res_win)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res_win); qqline(res_win)

res_win_skew <- skewness(res_win)
res_win_kurt <- kurtosis(res_win)
cat("Residual skewness:", round(res_win_skew, 3), "\n")
cat("Residual kurtosis:", round(res_win_kurt, 3), "\n")

# compare model with and without interaction

model_win_main <- lmer(
  log_tau_win ~ method + fr_s + lv_s + tau_ms_true_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)

anova(model_win_main, model_win)

# fit model with reml
model_win_reml <- lmer(
  log_tau_win ~ method * (fr_s + lv_s + tau_ms_true_s) 
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

# check residuals 
res_win_reml  <- residuals(model_win_reml)

ggplot(data.frame(res_win_reml), aes(x = res_win_reml)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res_win_reml); qqline(res_win_reml)

res_win_skew <- skewness(res_win_reml)
res_win_kurt <- kurtosis(res_win_reml)
cat("Residual skewness:", round(res_win_skew, 3), "\n")
cat("Residual kurtosis:", round(res_win_kurt, 3), "\n")

summary(model_win_reml)
confint(model_win_reml)


##################################################################
# For interactions: how the isttc/acf diff change 
##################################################################

# lv
a_min <- min(df$lv, na.rm=TRUE)
a_max <- max(df$lv, na.rm=TRUE)

sd_lv <- sd(df$lv, na.rm=TRUE)
delta_sd_full <- (a_max - a_min) / sd_lv

beta_lv <- fixef(model_win_reml)["methodisttc_full:lv_s"]

pct_full <- (10^(beta_lv * delta_sd_full) - 1) * 100
pct_full

# fr
a_min <- min(df$fr, na.rm=TRUE)
a_max <- max(df$fr, na.rm=TRUE)

sd_fr <- sd(df$fr, na.rm=TRUE)
delta_sd_full <- (a_max - a_min) / sd_fr

beta_fr <- fixef(model_win_reml)["methodisttc_full:fr_s"]

pct_full_fr <- (10^(beta_fr * delta_sd_full) - 1) * 100
pct_full_fr

# true tau
a_min <- min(df$tau_ms_true, na.rm=TRUE)
a_max <- max(df$tau_ms_true, na.rm=TRUE)

sd_tau <- sd(df$tau_ms_true, na.rm=TRUE)
delta_sd_full <- (a_max - a_min) / sd_tau

beta_tau <- fixef(model_win_reml)["methodisttc_full:tau_ms_true_s"]

pct_full_tau <- (10^(beta_tau * delta_sd_full) - 1) * 100
pct_full_tau


##################################################################
# For alpha: getting the crossover point
##################################################################

beta_meth   <- fixef(model_win_reml)["methodisttc_full"]
beta_methA  <- fixef(model_win_reml)["methodisttc_full:lv_s"]

# solve for alpha_s at which the predicted log10‐difference = 0
lv_s_cross <- -beta_meth / beta_methA

# get raw alpha
lv_mean <- mean(df$lv, na.rm=TRUE)
lv_sd   <- sd(df$lv,   na.rm=TRUE)
lv_cross <- lv_mean + lv_s_cross * lv_sd

lv_s_cross
lv_cross


##################################################################
# Calculate for min max for alpha, tau and fr (for all interactions)
##################################################################

# todo

##################################################################
#PLOTS
##################################################################


# forest plot of fixed effects - simple version
plot_model(
  model_win_reml, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)


# forest plot of fixed effects
fe <- tidy(model_win_reml, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "methodisttc_full" = "Method",
                       "fr_s" = "Firing rate (SD)",
                       "lv_s" = "Lv (SD)",
                       "tau_ms_true_s" = "True tau (SD)",
                       "methodisttc_full:fr_s" = "Method × FR",
                       "methodisttc_full:lv_s" = "Method × Lv",
                       "methodisttc_full:tau_ms_true_s" = "Method × True tau"),

    term = factor(term, levels = c(
      "Method",
      "Firing rate (SD)",
      "Lv (SD)",
      "True tau (SD)",
      "Method × FR",
      "Method × Lv",
      "Method × True tau"
    )),
    term = factor(term, levels = rev(levels(factor(term)))))

ggplot(fe, aes(x = term, y = ratio)) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high), width = 0.2) +
  geom_point(size = 3, color = "steelblue") +
  coord_flip() +
  scale_y_continuous(
    name = "Effect on tau_diff_rel"
  ) +
  labs(
    x     = NULL,
    title = "Fixed-Effects (Back-transformed from log10 to %)"
  ) + 
  theme_minimal(base_size = 14)


# Prediction plots, separate for fr, alpha and true_tau
# making new small grid because out of memory issues 

fr_grid <- seq(min(df$fr_s), max(df$fr_s), length.out = 20)
new_fr <- expand.grid(
  fr_s            = fr_grid,
  alpha_s         = 0,
  tau_ms_true_s   = 0,
  method          = levels(df$method)
)
X_fr    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_fr)

lv_grid <- seq(min(df$lv_s), max(df$lv_s), length.out = 20)
new_lv <- expand.grid(
  fr_s            = 0,
  lv_s         = lv_grid,
  tau_ms_true_s   = 0,
  method          = levels(df$method)
)
X_lv    <- model.matrix(~ method * (fr_s + lv_s + tau_ms_true_s), new_lv)

tau_grid <- seq(min(df$tau_ms_true_s), max(df$tau_ms_true_s), length.out = 20)
new_tau <- expand.grid(
  fr_s            = 0,
  alpha_s         = 0,
  tau_ms_true_s   = tau_grid,
  method          = levels(df$method)
)
X_tau    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_tau)


beta    <- fixef(model_win_reml)
Vb      <- vcov(model_win_reml)

# predicted log‐outcome and SE
new_fr <- new_fr %>%
  mutate(
    pred_log  = as.numeric(X_fr %*% beta),
    se_log    = sqrt(diag(X_fr %*% Vb %*% t(X_fr))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    fr = fr_s * sd(df$fr,   na.rm=TRUE) + mean(df$fr,   na.rm=TRUE)
  )
new_lv <- new_lv %>%
  mutate(
    pred_log = as.numeric(X_lv %*% beta),
    se_log   = sqrt(diag(X_lv %*% Vb %*% t(X_lv))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    lv    = lv_s * sd(df$lv, na.rm=TRUE) + mean(df$lv, na.rm=TRUE)
  )
new_tau <- new_tau %>%
  mutate(
    pred_log = as.numeric(X_tau %*% beta),
    se_log   = sqrt(diag(X_tau %*% Vb %*% t(X_tau))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    tau_ms_true    = tau_ms_true_s * sd(df$tau_ms_true, na.rm=TRUE) + mean(df$tau_ms_true, na.rm=TRUE)
  )

p1 <- ggplot(new_fr, aes(x = fr, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Firing rate (Hz)", y = "Predicted REE") +
  scale_y_continuous(
    breaks = log10(c(5, 10, 20)),
    labels = c("5", "10", "20")
  ) +
  scale_color_manual(values = c("#718190","#14a9e2")) +
  scale_fill_manual(values = c("#718190","#14a9e2")) +
  theme_minimal(base_size = 14)

p2 <- ggplot(new_lv, aes(x = lv, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Local variation (a.u.)", y = NULL) +
  scale_y_continuous(
    breaks = log10(c(1, 5, 10, 20, 40)),
    labels = c("1", "5", "10", "20", "40")
  ) +
  scale_color_manual(values = c("#718190","#14a9e2")) +
  scale_fill_manual(values = c("#718190","#14a9e2")) +
  theme_minimal(base_size = 14)

p3 <- ggplot(new_tau, aes(x = tau_ms_true, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "True tau", y = NULL) +
  scale_y_continuous(
    breaks = log10(c(11, 12, 13)),
    labels = c("11", "12", "13")
  ) +
  scale_color_manual(values = c("#718190","#14a9e2")) +
  scale_fill_manual(values = c("#718190","#14a9e2")) +
  theme_minimal(base_size = 14)

library(patchwork)
p1 | p2 | p3



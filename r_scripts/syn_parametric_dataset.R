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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_plot_long_df.csv", 
               stringsAsFactors = TRUE)
#summary(df)
#head(df)

# relevel
df$method <- relevel(df$method, ref = "acf_full")
# scale
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    alpha_s         = as.numeric(scale(alpha)),
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
  log_tau_diff ~ method * (fr_s + alpha_s + tau_ms_true_s)
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
  log_tau_win ~ method * (fr_s + alpha_s + tau_ms_true_s)
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
  log_tau_win ~ method + fr_s + alpha_s + tau_ms_true_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)

anova(model_win_main, model_win)

# fit model with reml
model_win_reml <- lmer(
  log_tau_win ~ method * (fr_s + alpha_s + tau_ms_true_s) 
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

##################################################################
#PLOTS
##################################################################

# forest plot of fixed effects
fe <- tidy(model_win_reml, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%             # drop the intercept
  mutate(
    ratio     = exp(estimate),                  # back‐transform point estimate
    ci.low    = exp(conf.low),                  # back‐transform lower CI
    ci.high   = exp(conf.high),                 # back‐transform upper CI
    term      = recode(term,                    # nicer labels
                       "methodisttc_full" = "Method: ist tc vs. acf",
                       "fr_s"               = "Firing rate (SD)",
                       "alpha_s"            = "Alpha (SD)",
                       "tau_ms_true_s"      = "True τ (SD)",
                       "methodisttc_full:fr_s"       = "Meth×FR",
                       "methodisttc_full:alpha_s"    = "Meth×Alpha",
                       "methodisttc_full:tau_ms_true_s" = "Meth×True τ")
  )

ggplot(fe, aes(x = fct_reorder(term, ratio), y = ratio)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high), width = 0.2) +
  geom_point(size = 3, color = "steelblue") +
  coord_flip() +
  scale_y_continuous(
    name = "Multiplicative effect on τ-diff\n(isttc_full vs. acf_full)",
    breaks = c(0.9, 1.0, 1.1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  labs(x = NULL,
       title = "Fixed-Effects (Back-transformed to Ratio Scale)") +
  theme_minimal(base_size = 14)


# more fancy forest plot
fe <- tidy(model_win_reml, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    ratio   = exp(estimate),
    ci.low  = exp(conf.low),
    ci.high = exp(conf.high),
    pct_lbl = percent(ratio, accuracy = 3),   # e.g. "96%"  
    term    = recode(term,
                     "methodisttc_full"              = "Method",
                     "fr_s"                           = "FR (SD)",
                     "alpha_s"                        = "Alpha (SD)",
                     "tau_ms_true_s"                  = "True τ (SD)",
                     "methodisttc_full:fr_s"          = "Meth × FR",
                     "methodisttc_full:alpha_s"       = "Meth × Alpha",
                     "methodisttc_full:tau_ms_true_s" = "Meth × True τ"
    ),
    term = factor(term, levels = c(
      "Method",
      "FR (SD)",
      "Alpha (SD)",
      "True τ (SD)",
      "Meth × FR",
      "Meth × Alpha",
      "Meth × True τ"
    ))
  )

ggplot(fe, aes(x = term, y = ratio)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high), width = 0.2) +
  geom_point(size = 3, color = "steelblue") +
  geom_text(aes(label = pct_lbl), 
            nudge_y = 0.01,           # tiny shift to the right
            hjust   = 0,              # left‐justify labels
            size    = 4) +            # label font size
  coord_flip() +
  scale_y_continuous(
    name   = "Multiplicative effect on τ-diff\n(isttc_full vs. acf_full)",
    limits = c(0.85, 1.1),
    breaks = c(0.9, 1.0, 1.1),
    labels = percent_format(accuracy = 1)
  ) +
  labs(
    x     = NULL,
    title = "Fixed-Effects (Back-transformed to Ratio Scale)"
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

alpha_grid <- seq(min(df$alpha_s), max(df$alpha_s), length.out = 20)
new_alpha <- expand.grid(
  fr_s            = 0,
  alpha_s         = alpha_grid,
  tau_ms_true_s   = 0,
  method          = levels(df$method)
)
X_al    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_alpha)

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
    # back‐transform to original outcome scale
    # tau_diff_hat     = exp(pred_log),
    # tau_diff_ci_low  = exp(ci_low),
    # tau_diff_ci_high = exp(ci_high),
    fr = fr_s * sd(df$fr,   na.rm=TRUE) + mean(df$fr,   na.rm=TRUE)
  )
new_alpha <- new_alpha %>%
  mutate(
    pred_log = as.numeric(X_al %*% beta),
    se_log   = sqrt(diag(X_al %*% Vb %*% t(X_al))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    alpha    = alpha_s * sd(df$alpha, na.rm=TRUE) + mean(df$alpha, na.rm=TRUE)
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
  labs(x = "Firing rate (Hz)", y = "Predicted log-τ-diff") +
  scale_color_manual(values = c("steelblue","firebrick")) +
  scale_fill_manual(values = c("steelblue","firebrick")) +
  theme_minimal(base_size = 14)

p2 <- ggplot(new_alpha, aes(x = alpha, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Excitation strength (a.u.)", y = NULL) +
  scale_color_manual(values = c("steelblue","firebrick")) +
  scale_fill_manual(values = c("steelblue","firebrick")) +
  theme_minimal(base_size = 14)

p3 <- ggplot(new_tau, aes(x = tau_ms_true, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "True tau", y = NULL) +
  scale_color_manual(values = c("steelblue","firebrick")) +
  scale_fill_manual(values = c("steelblue","firebrick")) +
  theme_minimal(base_size = 14)

library(patchwork)
p1 | p2 | p3



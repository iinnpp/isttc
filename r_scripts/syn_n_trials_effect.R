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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\fixed_params_var_trials\\tau_plot_long_df.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")
# scale
df <- df %>%
  mutate(
    n_trials_s = as.numeric(scale(n_trials)),
    trial_len_ms_s = as.numeric(scale(trial_len_ms))
  )

# fit model
model_log <- lmer(
  tau_diff_rel_log10 ~ method * (n_trials_s + trial_len_ms_s)
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

# Winsorizing

# 2.5% and 97.5% cutoffs
qs <- quantile(df$tau_diff_rel_log10, probs = c(0.025, 0.975), na.rm = TRUE)
lo <- qs[1]; hi <- qs[2]
# pull everything below lo up to lo, above hi down to hi
df <- df %>%
  mutate(
    log_tau_win = pmax(pmin(tau_diff_rel_log10, hi), lo)
  )

model_win <- lmer(
  log_tau_win ~ method * (n_trials_s + trial_len_ms_s)
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
  log_tau_win ~ method + n_trials_s + trial_len_ms_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)

anova(model_win_main, model_win)

# fit model with reml
model_win_reml <- lmer(
  log_tau_win ~ method * (n_trials_s + trial_len_ms_s) 
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
                       "methodsttc_trial_avg" = "STTC avg",
                       "methodsttc_trial_concat" = "STTC concat",
                       "n_trials_s" = "N trials",
                       "trial_len_ms_s" = "Trial len",
                       "methodsttc_trial_avg:n_trials_s" = "STTC avg x n trials",
                       "methodsttc_trial_concat:n_trials_s" = "STTC concat x n trials",
                       "methodsttc_trial_avg:trial_len_ms_s" = "STTC avg x trial len",
                       "methodsttc_trial_concat:trial_len_ms_s" = "STTC concat x trial len"),
    
    term = factor(term, levels = c(
      "STTC avg",
      "STTC concat",
      "N trials",
      "Trial len",
      "STTC avg x n trials",
      "STTC concat x n trials",
      "STTC avg x trial len",
      "STTC concat x trial len"
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

# Prediction plots, separate for trail_n and trail len
# making new small grid because out of memory issues 

n_trials_grid <- seq(min(df$n_trials_s), max(df$n_trials_s), length.out = 20)
new_n_trials <- expand.grid(
  n_trials_s      = n_trials_grid,
  trial_len_ms_s  = 0,
  method          = levels(df$method)
)
X_n_trials    <- model.matrix(~ method * (n_trials_s + trial_len_ms_s), new_n_trials)

trial_len_ms_grid <- seq(min(df$trial_len_ms_s), max(df$trial_len_ms_s), length.out = 20)
new_trial_len_ms <- expand.grid(
  n_trials_s      = 0,
  trial_len_ms_s  = trial_len_ms_grid,
  method          = levels(df$method)
)
X_trial_len_ms    <- model.matrix(~ method * (n_trials_s + trial_len_ms_s), new_trial_len_ms)


beta    <- fixef(model_win_reml)
Vb      <- vcov(model_win_reml)

# predicted log‐outcome and SE
new_n_trials <- new_n_trials %>%
  mutate(
    pred_log  = as.numeric(X_n_trials %*% beta),
    se_log    = sqrt(diag(X_n_trials %*% Vb %*% t(X_n_trials))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    n_trials = n_trials_s * sd(df$n_trials,   na.rm=TRUE) + mean(df$n_trials,   na.rm=TRUE)
  )
new_trial_len_ms <- new_trial_len_ms %>%
  mutate(
    pred_log = as.numeric(X_trial_len_ms %*% beta),
    se_log   = sqrt(diag(X_trial_len_ms %*% Vb %*% t(X_trial_len_ms))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    trial_len_ms    = trial_len_ms_s * sd(df$trial_len_ms, na.rm=TRUE) + mean(df$trial_len_ms, na.rm=TRUE)
  )

p1 <- ggplot(new_n_trials, aes(x = n_trials, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "N trials", y = "Predicted log-τ-diff") +
  scale_color_manual(values = c("steelblue","firebrick", "black")) +
  scale_fill_manual(values = c("steelblue","firebrick", "black")) +
  theme_minimal(base_size = 14)

p2 <- ggplot(new_trial_len_ms, aes(x = trial_len_ms, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Trial length (ms)", y = NULL) +
  scale_color_manual(values = c("steelblue","firebrick", "black")) +
  scale_fill_manual(values = c("steelblue","firebrick", "black")) +
  theme_minimal(base_size = 14)

library(patchwork)
p1 | p2 
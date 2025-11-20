library(sjPlot)
library(broom.mixed)    # for tidy()
library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(patchwork)  
library(forcats)
library(scales)

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau_comp_time\\trials\\var_trials\\compute_time_trials_long_df_1000_units.csv", 
               stringsAsFactors = TRUE)

df$method <- factor(df$method)
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")

df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    alpha_s         = as.numeric(scale(alpha)),
    tau_ms_true_s   = as.numeric(scale(tau_ms_true)),
    duration_s_s    = as.numeric(scale(n_trials))
  )

# ==========================================================
# 1. Baseline model: method only
# ==========================================================
m0 <- lmer(
  total_time_sec ~ method + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m0)
confint(m0, method="Wald")

# ==========================================================
# 2. FR, alpha, tau, duration (no interactions)
# ==========================================================
m1 <- lmer(
  total_time_sec ~ method + fr_s + alpha_s + tau_ms_true_s + duration_s_s +
    (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m1)
anova(m0, m1)

# ==========================================================
# 3. Full interaction model
# ==========================================================
m2 <- lmer(
  total_time_sec ~ method * (fr_s + alpha_s + tau_ms_true_s + duration_s_s) +
    (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m2)
anova(m1, m2)

# ==========================================================
# 4. Final model refit with REML
# ==========================================================
m2_reml <- lmer(
  total_time_sec ~ method * (fr_s + alpha_s + tau_ms_true_s + duration_s_s) +
    (1 | unit_id),
  data = df,
  REML = TRUE
)

summary(m2_reml)
confint(m2_reml)


## =========================================================
## PLOTS
## =========================================================


# forest plot of fixed effects - simple version
p <- plot_model(
  m2_reml, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.01, 0.02))
p10

# Prediction plots
# making new small grid because out of memory issues 

duration_grid <- seq(min(df$duration_s_s), max(df$duration_s_s), length.out = 20)
new_duration <- expand.grid(
  duration_s_s   = duration_grid,
  method          = levels(df$method)
)
new_duration <- new_duration %>%
  mutate(
    fr_s = 0,
    alpha_s = 0,
    tau_ms_true_s = 0
  )

X_duration    <- model.matrix(~ method * (duration_s_s + fr_s + alpha_s + tau_ms_true_s), new_duration)


beta    <- fixef(m2_reml)
Vb      <- vcov(m2_reml)

# predicted log‐outcome and SE
new_duration <- new_duration %>%
  mutate(
    pred_log  = as.numeric(X_duration %*% beta),
    se_log    = sqrt(diag(X_duration %*% Vb %*% t(X_duration))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    n_trials = duration_s_s * sd(df$n_trials,   na.rm=TRUE) + mean(df$n_trials,   na.rm=TRUE)
  )


p1 <- ggplot(new_duration, aes(x = n_trials, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Duration (sec)", y = "Predicted total_time_sec") +
#  scale_y_continuous(
#    breaks = log10(c(10, 15, 20, 25, 30)),
#    labels = c("10", "15", "20", "25", "30")
#  ) +
#  scale_x_continuous(
#    breaks = c(60, 150, 300, 450, 600),
#    labels = c("60", "150", "300", "450", "600")
#  ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p1 
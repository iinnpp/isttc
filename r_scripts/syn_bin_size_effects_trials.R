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

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\bin_size_runs\\trials\\summary_tau_trials_long_lags_df.csv", 
               stringsAsFactors = TRUE)


df$bin_size <- factor(df$bin_size)
df$bin_size <- relevel(df$bin_size, ref = "50")

df$method <- relevel(df$method, ref = "pearsonr_trial_avg")

df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    alpha_s         = as.numeric(scale(alpha)),
    tau_ms_true_s   = as.numeric(scale(tau_ms_true))
  )

df <- df %>%
  mutate(
    log_tau_ms = log10(tau_ms)
  )

# ==========================================================
# 1. Baseline model: bin_size only
# ==========================================================
m0 <- lmer(
  log_tau_ms ~ bin_size + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m0)
confint(m0, method="Wald")

# ==========================================================
# 2. FR, alpha, tau (no interactions)
# ==========================================================
m1 <- lmer(
  log_tau_ms ~ bin_size + method + fr_s + alpha_s + tau_ms_true_s + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m1)
anova(m0, m1)


# ==========================================================
# 3. Full interaction model
# ==========================================================
m2 <- lmer(
  log_tau_ms ~ bin_size * (method + fr_s + alpha_s + tau_ms_true_s) + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m2)
anova(m1, m2)


# ==========================================================
# 4. Final model refit with REML
# ==========================================================
m2_reml <- lmer(
  log_tau_ms ~ bin_size * (method + fr_s + alpha_s + tau_ms_true_s) + (1 | unit_id),
  data = df,
  REML = TRUE
)

summary(m2_reml)
#confint(m2_reml)


## =========================================================
## PLOTS
## =========================================================


# forest plot of fixed effects - simple version
p <- plot_model(
  m2_reml, 
  terms = c("bin_size10", "bin_size40", "bin_size60", "bin_size100", "bin_size10:methodsttc_trial_concat", "bin_size40:methodsttc_trial_concat", "bin_size60:methodsttc_trial_concat", "bin_size100:methodsttc_trial_concat"),
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.2, 0.1))
p10

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

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau_comp_time\\trials\\compute_time_trials_long_df_all_units.csv", 
               stringsAsFactors = TRUE)

df$method <- factor(df$method)
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")


# sanity check
str(df)
table(df$method)
summary(df$total_time_sec)
#summary(df$total_time_sec)


## =========================================================
## 1. Simplest model: method effect with random unit intercept
## =========================================================
# Model m0: only method + random intercept for unit
m0 <- lmer(
  total_time_sec ~ method + (1 | unit_id),
  data = df,
  REML = FALSE
)

summary(m0)
confint(m0, method = "Wald")


## =========================================================
## 2. Model with fr, alpha, tau_ms_true
## =========================================================
df <- df %>%
  mutate(
    fr_s          = as.numeric(scale(fr)),
    alpha_s       = as.numeric(scale(alpha)),
    tau_ms_true_s = as.numeric(scale(tau_ms_true))
  )

# Model m1: method + covariates, no interactions 
m1 <- lmer(
  total_time_sec ~ method + fr_s + alpha_s + tau_ms_true_s + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m1)
# fit is better!
anova(m0, m1)


## =========================================================
## 3. Model with interactions
## =========================================================
m2 <- lmer(
  total_time_sec ~ method * fr_s +
    method * alpha_s +
    method * tau_ms_true_s +
    (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(m2)
# the best fit!
anova(m1, m2)

# rerun with reml
m2_reml <- lmer(
  total_time_sec ~ method * fr_s +
    method * alpha_s +
    method * tau_ms_true_s +
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
p
p10 <- p + scale_y_continuous(limits = c(-0.03, 0.03))
p10









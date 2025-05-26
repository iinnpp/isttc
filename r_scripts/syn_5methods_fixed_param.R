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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\fixed_params\\tau_5methods_100trial_impl.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "acf_full")

# log‐transform, check tails - kurtosis still high
df <- df %>%
  mutate(
    log_tau_diff = log10(tau_diff_rel)
  )

model_log <- lmer(
  log_tau_diff ~ method + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(model_log)

plot_model(
  model_log, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
library(sjPlot)
library(broom.mixed)    # for tidy()
library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
#library(moments)
library(ggeffects)
#library(robustlmm)
#library(glmmTMB)
library(patchwork)  
library(forcats)
library(scales)

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\bin_size_runs\\full_signal\\summary_tau_full_long_acf_lags_df.csv", 
               stringsAsFactors = TRUE)

df$n_lags <- factor(df$n_lags)
df$n_lags <- relevel(df$n_lags, ref = "20")

# log‐transfom
df <- df %>%
  mutate(
    log_tau_ms = log10(tau_ms)
  )

df <- df %>%
  mutate(
    log_tau_diff_rel = log10(tau_diff_rel)
  )


model_log <- lmer(
  log_tau_ms ~ n_lags + (1 | unit_id),
  data = df,
  REML = FALSE
)

summary(model_log)

p <- plot_model(
  model_log, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.025, 0.025))
p10


model_log_ree <- lmer(
  log_tau_diff_rel ~ n_lags + (1 | unit_id),
  data = df,
  REML = FALSE
)
summary(model_log_ree)

p <- plot_model(
  model_log_ree, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.25, 0.25))
p10
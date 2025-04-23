library(rio)
library(sjPlot)
library(emmeans)
library(lme4)

df <- import("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\allen_mice\\dataset\\cut_30min\\summary_taus_plot_5methods_100_50_20_1_resampling_df_long.csv")
df$method <- factor(df$method, levels = c("isttc_full", "acf_full", "pearsonr", "sttc_avg", "sttc_concat"))
df$n_sampling <- factor(df$n_sampling, levels = c("1","20", "50", "100"))
df$method <- relevel(df$method, ref='isttc_full')
df$n_sampling <- relevel(df$n_sampling, ref='20')

# trials 100 sampling, control is isttc_full
df_5 <- df[df$n_sampling == "1", ]

model_5 <- lmer(tau_ms_log10 ~ method + (1 | unit_id), data=df_5)
summary(model_5)
confint(model_5)
emmeans(model_5, pairwise ~ method)

plot_model(
  model_5, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)

# also try MI
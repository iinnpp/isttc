Sys.setenv(LANG = "en")
rm(list  =  ls())


#library('performance')
library('sjPlot')
library('lme4')
library('rio')
library('emmeans')
library('ggplot2')
#library('lmerTest')

df <- import("E:/projects_q_30_10_2024/isttc/results/monkey/fixation_period_1000ms/stats/tau_average_trial_sttc_and_pearson_constrained_dataset_with_empty_0_1000_51padding_df.csv")

df$metric <- factor(df$metric)
df$area_unit_id <- factor(df$area_unit_id)

# fit the model
df$metric <- relevel(df$metric, ref='pearson')
model_lin_v1 <- lmer(tau_ms_log10 ~ metric + (1 | area_unit_id), data=df)
summary(model_lin_v1)
confint(model_lin_v1)

plot_model(model_lin_v1, type='pred', terms=c('metric'), ci.lvl = .95)

plot_model(
  model_lin_v1, 
  show.values = TRUE,
  value.offset = .4,
  value.size = 4,
  dot.size = 2,
  line.size = 0.4,
  vline.color = "blue",
  vline.size = 0.1,
  width = 0.2
) 

#+ scale_y_continuous(limits = c(0.0, 0.05))


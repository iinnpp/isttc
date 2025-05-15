# library(sjPlot)
# library(broom.mixed)    # for tidy()
# library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
# library(ggplot2)
# library(moments)
# library(ggeffects)
# library(robustlmm)
# library(glmmTMB)
# library(patchwork)  
# library(forcats)
# library(scales)


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\monkey\\fixation_period_1000ms_no_empty\\acf_tau_full_df.csv", 
               stringsAsFactors = TRUE)

# Drop rows with NaN 
df_clean <- subset(df, !is.na(tau_ms_log_10))

m1 <- lmer(tau_ms_log_10 ~ method + (1 | unit_id), data = df_clean)

summary(m1)
qqnorm(resid(m1)); qqline(resid(m1))  


car::Anova(m1, type = 3)     
emm <- emmeans(m1, "method")
pairs(emm, adjust = "tukey")
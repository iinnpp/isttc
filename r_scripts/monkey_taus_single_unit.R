library(sjPlot)
# library(broom.mixed)    # for tidy()
library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
library(effects)
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
df_clean <- subset(
  df,
  !is.na(tau_ms_log_10) &
    decline_150_250 == "True"
)

m1 <- lmer(tau_ms_log_10 ~ method + (1 | unit_id), data = df_clean)

summary(m1)
qqnorm(resid(m1)); qqline(resid(m1))  


car::Anova(m1, type = 3)     
emm <- emmeans(m1, "method")
pairs(emm, adjust = "tukey")


# marginal means
emm_df <- as.data.frame(emm)

ggplot(emm_df, aes(x = method, y = emmean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(
    ymin = emmean - SE,
    ymax = emmean + SE
  ), width = 0.2) +
  labs(
    x = "Method",
    y = "Estimated log10 IT (ms)"
  ) +
  theme_classic(base_size = 14)

# forest plot of effects
plot_model(
  m1, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)


# plot  pairwise contrasts with CIs
pc <- pairs(emm, adjust = "tukey")
pc_df <- summary(pc, infer = c(TRUE, TRUE)) %>% as.data.frame()

ggplot(pc_df, aes(x = estimate, y = contrast)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(
    x = expression("Tau(ms)"),
    y = NULL,
    title = "Pairwise contrasts"
  ) +
  theme_minimal(base_size = 14)
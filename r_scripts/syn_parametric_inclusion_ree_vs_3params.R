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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_4methods_long_all_not_nan_r2_not_neg_df.csv", 
               stringsAsFactors = TRUE)


df <- df %>%
  mutate(
    log_tau_diff = log10(tau_diff_rel)
  )


#######################
##### MODELS #####

#######################
##### ACF decline #####

# relevel
df$acf_decline <- relevel(df$acf_decline, ref = "False")

model_acf_decline <- lmer(
  log_tau_diff ~ acf_decline + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_acf_decline)
confint(model_acf_decline)


#######################
##### CI 0 ############
# relevel
df$ci_zero_excluded <- factor(df$ci_zero_excluded)
df$ci_zero_excluded <- relevel(df$ci_zero_excluded, ref = "0")

model_ci <- lmer(
  log_tau_diff ~ ci_zero_excluded + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_ci)
confint(model_ci)


#########################
##### fit_r_squared #####
df_pos_r2 <- df[df$fit_r_squared > 0, ]
df_pos_r2 <- df_pos_r2 %>%
  mutate(
    log_fit_r_squared = log10(fit_r_squared)
  )

model_r_squared <- lmer(
  log_tau_diff ~ log_fit_r_squared + (1 | unit_id),
  data = df_pos_r2,
  REML = TRUE
)
summary(model_r_squared)
confint(model_r_squared)

model_r_squared_non_neg <- lmer(
  log_tau_diff ~ fit_r_squared + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_r_squared_non_neg)
confint(model_r_squared_non_neg)

#########################
##### PLOTS #####

# predicted values for r2

# this gives out of memory
# plot_model(model_r_squared, type='pred', terms='log_fit_r_squared', ci.lvl = .68)


# plotting with new grid
grid_r2 <- seq(min(df$fit_r_squared), max(df$fit_r_squared), length.out = 100)
new_data <- data.frame(
  fit_r_squared = grid_r2,
  unit_id = NA  
)

X_new <- model.matrix(~ fit_r_squared, new_data)
beta <- fixef(model_r_squared_non_neg)
Vb   <- vcov(model_r_squared_non_neg)
new_data <- new_data %>%
  mutate(
    pred_log  = as.numeric(X_new %*% beta),
    se_log    = sqrt(diag(X_new %*% Vb %*% t(X_new))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log
  )


p <- ggplot(new_data, aes(x = fit_r_squared, y = pred_log)) +
  geom_line(color = "#00A9E2", size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, fill = "#00A9E2") +
  labs(x = "R-squared (a.u.)", y = "Predicted REE") +
    scale_y_continuous(
      limits = c(1, 3),
      breaks = log10(c(10, 100, 1000)),
      labels = c("10", "100", "1000")
  ) +
  theme_minimal(base_size = 14)

p










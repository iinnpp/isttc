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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_plot_all_long_not_nan_df.csv", 
               stringsAsFactors = TRUE)
trials_df <- droplevels(df[df$method %in% c("acf_full", "isttc_full"), ])


#summary(df)
#head(df)

# relevel
trials_df$method <- relevel(trials_df$method, ref = "acf_full")
# scale
trials_df <- trials_df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    alpha_s         = as.numeric(scale(alpha)),
    tau_ms_true_s   = as.numeric(scale(tau_ms_true))
  )



model_non_log <- lmer(
  fit_r_squared ~ method * (fr_s + alpha_s + tau_ms_true_s)
  + (1 | unit_id),
  data = trials_df,
  REML = FALSE
)

summary(model_non_log)

# check residuals - heavy tail present!
res  <- residuals(model_non_log)

ggplot(data.frame(res), aes(x = res)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res); qqline(res)

res_skew <- skewness(res)
res_kurt <- kurtosis(res)
cat("Residual skewness:", round(res_skew, 3), "\n")
cat("Residual kurtosis:", round(res_kurt, 3), "\n")

### Plots ###
# forest plot of fixed effects - simple version
p <- plot_model(
  model_non_log, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.1, 0.25))
p10

# Prediction plots, separate for fr, alpha and true_tau
# making new small grid because out of memory issues 

fr_grid <- seq(min(trials_df$fr_s), max(trials_df$fr_s), length.out = 20)
new_fr <- expand.grid(
  fr_s            = fr_grid,
  alpha_s         = 0,
  tau_ms_true_s   = 0,
  method          = levels(trials_df$method)
)
X_fr    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_fr)

alpha_grid <- seq(min(trials_df$alpha_s), max(trials_df$alpha_s), length.out = 20)
new_alpha <- expand.grid(
  fr_s            = 0,
  alpha_s         = alpha_grid,
  tau_ms_true_s   = 0,
  method          = levels(trials_df$method)
)
X_al    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_alpha)

tau_grid <- seq(min(trials_df$tau_ms_true_s), max(trials_df$tau_ms_true_s), length.out = 20)
new_tau <- expand.grid(
  fr_s            = 0,
  alpha_s         = 0,
  tau_ms_true_s   = tau_grid,
  method          = levels(trials_df$method)
)
X_tau    <- model.matrix(~ method * (fr_s + alpha_s + tau_ms_true_s), new_tau)


beta    <- fixef(model_non_log)
Vb      <- vcov(model_non_log)

# predicted log‐outcome and SE
new_fr <- new_fr %>%
  mutate(
    pred_log  = as.numeric(X_fr %*% beta),
    se_log    = sqrt(diag(X_fr %*% Vb %*% t(X_fr))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    fr = fr_s * sd(trials_df$fr,   na.rm=TRUE) + mean(trials_df$fr,   na.rm=TRUE)
  )
new_alpha <- new_alpha %>%
  mutate(
    pred_log = as.numeric(X_al %*% beta),
    se_log   = sqrt(diag(X_al %*% Vb %*% t(X_al))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    alpha    = alpha_s * sd(trials_df$alpha, na.rm=TRUE) + mean(trials_df$alpha, na.rm=TRUE)
  )
new_tau <- new_tau %>%
  mutate(
    pred_log = as.numeric(X_tau %*% beta),
    se_log   = sqrt(diag(X_tau %*% Vb %*% t(X_tau))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    tau_ms_true    = tau_ms_true_s * sd(trials_df$tau_ms_true, na.rm=TRUE) + mean(trials_df$tau_ms_true, na.rm=TRUE)
  )

p1 <- ggplot(new_fr, aes(x = fr, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Firing rate (Hz)", y = "Predicted REE") +
  # scale_y_continuous(
  #   breaks = log10(c(5, 10, 15, 20)),
  #   labels = c("5", "10", "15", "20")
  # ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p2 <- ggplot(new_alpha, aes(x = alpha, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Excitation strength (a.u.)", y = NULL) +
  # scale_y_continuous(
  #   breaks = log10(c(5, 10, 15, 20)),
  #   labels = c("5", "10", "15", "20")
  # ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p3 <- ggplot(new_tau, aes(x = tau_ms_true, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "True tau", y = NULL) +
  # scale_y_continuous(
  #   breaks = log10(c(5, 10, 15, 20)),
  #   labels = c("5", "10", "15", "20")
  # ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

library(patchwork)
p1 | p2 | p3



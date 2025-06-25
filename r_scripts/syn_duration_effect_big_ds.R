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


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_plot_long_var_len_df.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "acf_full")
# scale
df <- df %>%
  mutate(
    duration_ms_s = as.numeric(scale(duration_s))
  )
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr)),
    alpha_s         = as.numeric(scale(alpha)),
    tau_ms_true_s   = as.numeric(scale(tau_ms_true))
  )

# log‐transfom
df <- df %>%
  mutate(
    log_tau_diff = log10(tau_diff_rel)
  )


# fit model
model_log <- lmer(
  log_tau_diff ~ method * duration_ms_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)
# check residuals - heavy tail present!
res  <- residuals(model_log)

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

# Winsorizing

# 2.5% and 97.5% cutoffs
qs <- quantile(df$log_tau_diff, probs = c(0.025, 0.975), na.rm = TRUE)
lo <- qs[1]; hi <- qs[2]
# pull everything below lo up to lo, above hi down to hi
df <- df %>%
  mutate(
    log_tau_win = pmax(pmin(log_tau_diff, hi), lo)
  )

model_win <- lmer(
  log_tau_win ~ method * duration_ms_s
  + (1 | unit_id),
  data   = df,
  REML   = FALSE
)

summary(model_win)

# check residuals 
res_win  <- residuals(model_win)

ggplot(data.frame(res_win), aes(x = res_win)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res_win); qqline(res_win)

res_win_skew <- skewness(res_win)
res_win_kurt <- kurtosis(res_win)
cat("Residual skewness:", round(res_win_skew, 3), "\n")
cat("Residual kurtosis:", round(res_win_kurt, 3), "\n")

# compare model with and without interaction

model_win_main <- lmer(
  log_tau_win ~ method + duration_ms_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)

anova(model_win_main, model_win)

# fit model with reml
model_win_reml <- lmer(
  log_tau_win ~ method * (duration_ms_s + fr_s + alpha_s + tau_ms_true_s)
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

# check residuals 
res_win_reml  <- residuals(model_win_reml)

ggplot(data.frame(res_win_reml), aes(x = res_win_reml)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "black") +
  geom_density() +
  labs(title = "Residuals: Histogram & Density",
       x = "Residual", y = "Density")

qqnorm(res_win_reml); qqline(res_win_reml)

res_win_skew <- skewness(res_win_reml)
res_win_kurt <- kurtosis(res_win_reml)
cat("Residual skewness:", round(res_win_skew, 3), "\n")
cat("Residual kurtosis:", round(res_win_kurt, 3), "\n")

summary(model_win_reml)


##################################################################
#PLOTS
##################################################################


# forest plot of fixed effects - simple version
p <- plot_model(
  model_win_reml, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-0.2, 0.05))
p10 + coord_flip()


# forest plot of fixed effects - back scaled
fe <- tidy(model_win_reml, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "methodisttc_full" = "Method",
                       "duration_ms_s" = "Duration",
                       "fr_s" = "Firing rate (SD)",
                       "alpha_s" = "Alpha (SD)",
                       "tau_ms_true_s" = "True tau (SD)",
                       "methodisttc_full:duration_ms_s" = "Method × Duration",
                       "methodisttc_full:fr_s" = "Method × FR",
                       "methodisttc_full:alpha_s" = "Method × Alpha",
                       "methodisttc_full:tau_ms_true_s" = "Method × True tau"),
    
    term = factor(term, levels = c(
      "Method",
      "Duration",
      "Firing rate (SD)",
      "Alpha (SD)",
      "True tau (SD)",
      "Method × Duration",
      "Method × FR",
      "Method × Alpha",
      "Method × True tau"
    )),
    term = factor(term, levels = rev(levels(factor(term)))))

ggplot(fe, aes(x = term, y = ratio)) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high), width = 0.2) +
  geom_point(size = 3, color = "steelblue") +
  coord_flip() +
  scale_y_continuous(
    name = "Effect on tau_diff_rel"
  ) +
  labs(
    x     = NULL,
    title = "Fixed-Effects (Back-transformed from log10 to %)"
  ) + 
  theme_minimal(base_size = 14)


# Prediction plots
# making new small grid because out of memory issues 

duration_grid <- seq(min(df$duration_ms_s), max(df$duration_ms_s), length.out = 20)
new_duration <- expand.grid(
  duration_ms_s   = duration_grid,
  method          = levels(df$method)
)
X_duration    <- model.matrix(~ method * (duration_ms_s), new_duration)


beta    <- fixef(model_win_reml)
Vb      <- vcov(model_win_reml)

# predicted log‐outcome and SE
new_duration <- new_duration %>%
  mutate(
    pred_log  = as.numeric(X_duration %*% beta),
    se_log    = sqrt(diag(X_duration %*% Vb %*% t(X_duration))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    duration_s = duration_ms_s * sd(df$duration_s,   na.rm=TRUE) + mean(df$duration_s,   na.rm=TRUE)
  )


p1 <- ggplot(new_duration, aes(x = duration_s, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Duration (sec)", y = "Predicted log tau diff") +
  scale_color_manual(values = c("#708090","#00A9E2")) +
  scale_fill_manual(values = c("#708090","#00A9E2")) +
  theme_minimal(base_size = 14)

p1 



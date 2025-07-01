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

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_plot_long_trials_parametric_df.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")
# scale
df <- df %>%
  mutate(
    n_trials_s = as.numeric(scale(n_trials))
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
  log_tau_diff ~ method * n_trials_s
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
qs <- quantile(df$log_tau_diff, probs = c(0.05, 0.95), na.rm = TRUE)
lo <- qs[1]; hi <- qs[2]
# pull everything below lo up to lo, above hi down to hi
df <- df %>%
  mutate(
    log_tau_win = pmax(pmin(log_tau_diff, hi), lo)
  )

model_win <- lmer(
  log_tau_win ~ method * n_trials_s
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
  log_tau_win ~ method + n_trials_s
  + (1 | unit_id),
  data = df,
  REML = FALSE
)

anova(model_win_main, model_win)

# fit model with reml
model_win_reml <- lmer(
  log_tau_win ~ method * (n_trials_s + fr_s + alpha_s + tau_ms_true_s)
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
confint(model_win_reml)


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
p10 <- p + scale_y_continuous(limits = c(-1, 1))
p10 + coord_flip()


# forest plot of fixed effects - back scaled
fe <- tidy(model_win_reml, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "methodsttc_trial_concat" = "Pearsonr vs STTC concat",
                       "n_trials_s" = "N trials",
                       "fr_s" = "Firing rate (SD)",
                       "alpha_s" = "Alpha (SD)",
                       "tau_ms_true_s" = "True tau (SD)",
                       "methodsttc_trial_concat:n_trials_s" = "STTC concat × N trials",
                       "methodsttc_trial_concat:fr_s" = "STTC concat × FR",
                       "methodsttc_trial_concat:alpha_s" = "STTC concat × Alpha",
                       "methodsttc_trial_concat:tau_ms_true_s" = "STTC concat × True tau"),
    
    term = factor(term, levels = c(
      "Pearsonr vs STTC concat",
      "N trials",
      "Firing rate (SD)",
      "Alpha (SD)",
      "True tau (SD)",
      "STTC concat × N trials",
      "STTC concat × FR",
      "STTC concat × Alpha",
      "STTC concat × True tau"
      
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

duration_grid <- seq(min(df$n_trials_s), max(df$n_trials_s), length.out = 20)
new_duration <- expand.grid(
  n_trials_s   = duration_grid,
  method          = levels(df$method)
)

new_duration <- new_duration %>%
  mutate(
    fr_s = 0,
    alpha_s = 0,
    tau_ms_true_s = 0
  )

X_duration    <- model.matrix(~ method * (n_trials_s + fr_s + alpha_s + tau_ms_true_s), new_duration)


beta    <- fixef(model_win_reml)
Vb      <- vcov(model_win_reml)

# predicted log‐outcome and SE
new_duration <- new_duration %>%
  mutate(
    pred_log  = as.numeric(X_duration %*% beta),
    se_log    = sqrt(diag(X_duration %*% Vb %*% t(X_duration))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    n_trials = n_trials_s * sd(df$n_trials,   na.rm=TRUE) + mean(df$n_trials,   na.rm=TRUE)
  )


p1 <- ggplot(new_duration, aes(x = n_trials, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "N trials", y = "Predicted log tau diff") +
  scale_y_continuous(
    breaks = log10(c(60, 80, 100, 120, 140, 160)),
    labels = c("60", "80", "100", "120", "140", "160"),
    limits = log10(c(60, 160))
  ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p1 



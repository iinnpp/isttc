library(sjPlot)
library(broom.mixed)    # for tidy()
library(dplyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
library(moments)
library(ggeffects)
library(patchwork)  
library(forcats)
library(scales)


df <- read.csv("D:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\mice\\dataset\\cut_30min\\tau_long_2trial_methods_df.csv", 
               stringsAsFactors = TRUE)
# df <- df %>% filter(fr_hz_spont_30min <= 10)
#summary(df)
#head(df)

# relevel
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")
# scale
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr_hz_spont_30min)),
    alpha_s         = as.numeric(scale(lv))
  )

# log‐transform, check tails - kurtosis still high
df <- df %>%
  mutate(
    log_tau_diff = log10(tau_diff_rel)
  )

# fit model

model_log <- lmer(
  log_tau_diff ~ method * (fr_s + alpha_s)
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

summary(model_log)

# compare model with and without interaction

model_main <- lmer(
  log_tau_win ~ method + fr_s + alpha_s
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

anova(model_main, model_log)

summary(model_main)


##################################################################
#PLOTS
##################################################################


# forest plot of fixed effects - simple version
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


# forest plot of fixed effects
fe <- tidy(model_log, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "methodsttc_trial_concat" = "Method",
                       "fr_s" = "Firing rate (SD)",
                       "alpha_s" = "Alpha (SD)",
                       "methodsttc_trial_concat:fr_s" = "Method × FR",
                       "methodsttc_trial_concat:alpha_s" = "Method × Alpha"),
    
    term = factor(term, levels = c(
      "Method",
      "Firing rate (SD)",
      "Alpha (SD)",
      "Method × FR",
      "Method × Alpha"
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


# Prediction plots, separate for fr, alpha and true_tau
# making new small grid because out of memory issues 

fr_grid <- seq(min(df$fr_s), max(df$fr_s), length.out = 100)
new_fr <- expand.grid(
  fr_s            = fr_grid,
  alpha_s         = 0,
  tau_ms_true_s   = 0,
  method          = levels(df$method)
)
X_fr    <- model.matrix(~ method * (fr_s + alpha_s), new_fr)

alpha_grid <- seq(min(df$alpha_s), max(df$alpha_s), length.out = 100)
new_alpha <- expand.grid(
  fr_s            = 0,
  alpha_s         = alpha_grid,
  tau_ms_true_s   = 0,
  method          = levels(df$method)
)
X_al    <- model.matrix(~ method * (fr_s + alpha_s), new_alpha)


beta    <- fixef(model_log)
Vb      <- vcov(model_log)

# predicted log‐outcome and SE
new_fr <- new_fr %>%
  mutate(
    pred_log  = as.numeric(X_fr %*% beta),
    se_log    = sqrt(diag(X_fr %*% Vb %*% t(X_fr))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    # back‐transform to original outcome scale
    # tau_diff_hat     = exp(pred_log),
    # tau_diff_ci_low  = exp(ci_low),
    # tau_diff_ci_high = exp(ci_high),
    fr = fr_s * sd(df$fr_hz_spont_30min,   na.rm=TRUE) + mean(df$fr_hz_spont_30min,   na.rm=TRUE)
  )
new_alpha <- new_alpha %>%
  mutate(
    pred_log = as.numeric(X_al %*% beta),
    se_log   = sqrt(diag(X_al %*% Vb %*% t(X_al))),
    ci_low   = pred_log - 1.96 * se_log,
    ci_high  = pred_log + 1.96 * se_log,
    alpha    = alpha_s * sd(df$lv, na.rm=TRUE) + mean(df$lv, na.rm=TRUE)
  )

p1 <- ggplot(new_fr, aes(x = fr, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Firing rate (Hz)", y = "Predicted log-τ-diff") +
  scale_y_continuous(
    breaks = log10(c(100, 250, 500, 750, 1000)),
    labels = c("100", "250", "500", "750", "1000")
  ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p2 <- ggplot(new_alpha, aes(x = alpha, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Excitation strength (a.u.)", y = NULL) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

library(patchwork)
p1 | p2 


plot_model(model_log,
           type = "pred",
           terms = c("fr_s", "method"),  # marginal effect of fr_s, grouped by method
           ci.lvl = 0.95,
           title = "Marginal Effect of Firing Rate",
           axis.title = c("Firing rate (scaled)", "Predicted log-τ-diff"),
           colors = c("#f4a91c", "#955da2"))  # adjust colors

plot_model(model_log,
           type = "pred",
           terms = c("alpha_s", "method"),  # marginal effect of alpha_s, grouped by method
           ci.lvl = 0.95,
           title = "Marginal Effect of Excitation Strength",
           axis.title = c("Excitation strength (scaled)", "Predicted log-τ-diff"),
           colors = c("#f4a91c", "#955da2"))


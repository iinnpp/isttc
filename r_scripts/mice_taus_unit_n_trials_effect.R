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


df <- read.csv("D:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\mice\\dataset\\tau_unit_long_2_methods_var_trials_df.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "pearsonr_trial_avg")

df <- df %>%
  mutate(
    len_s = as.numeric(scale(n_trials))
  )

# scale
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr_hz_spont_30min)),
    alpha_s         = as.numeric(scale(lv))
  )

# fit model

model_log <- lmer(
  tau_diff_rel_log10 ~ method * (fr_s + alpha_s + len_s)
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

summary(model_log)

# forest plot of fixed effects - simple version
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
p10 <- p + scale_y_continuous(limits = c(-0.175, 0.15))
p10 + coord_flip()


# forest plot of fixed effects - back scaled
fe <- tidy(model_log, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "methodisttc_trial_concat" = "Method",
                       "len_s" = "Duration",
                       "fr_s" = "Firing rate (SD)",
                       "alpha_s" = "Alpha (SD)",
                       "methodisttc_trial_concat:len_s" = "Method × Duration",
                       "methodisttc_trial_concat:fr_s" = "Method × FR",
                       "methodisttc_trial_concat:alpha_s" = "Method × Alpha"),
    
    term = factor(term, levels = c(
      "Method",
      "Duration",
      "Firing rate (SD)",
      "Alpha (SD)",
      "Method × Duration",
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


# Prediction plots
# making new small grid because out of memory issues 

duration_grid <- seq(min(df$len_s), max(df$len_s), length.out = 20)
new_duration <- expand.grid(
  len_s   = duration_grid,
  method          = levels(df$method)
)
new_duration <- new_duration %>%
  mutate(
    fr_s = 0,
    alpha_s = 0,
  )

X_duration    <- model.matrix(~ method * (len_s + fr_s + alpha_s), new_duration)


beta    <- fixef(model_log)
Vb      <- vcov(model_log)

# predicted log‐outcome and SE
new_duration <- new_duration %>%
  mutate(
    pred_log  = as.numeric(X_duration %*% beta),
    se_log    = sqrt(diag(X_duration %*% Vb %*% t(X_duration))),
    ci_low    = pred_log - 1.96 * se_log,
    ci_high   = pred_log + 1.96 * se_log,
    n_trials = len_s * sd(df$n_trials,   na.rm=TRUE) + mean(df$n_trials,   na.rm=TRUE)
  )


p1 <- ggplot(new_duration, aes(x = n_trials, y = pred_log, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.2, color = NA) +
  labs(x = "Duration (sec)", y = "Predicted log tau diff") +
  scale_y_continuous(
    breaks = log10(c(150, 200, 250, 300)),
    labels = c("150", "200", "250", "300")
    #  ) +
    #  scale_x_continuous(
    #    breaks = c(60, 150, 300, 450, 600),
    #    labels = c("60", "150", "300", "450", "600")
  ) +
  scale_color_manual(values = c("#f4a91c","#955da2")) +
  scale_fill_manual(values = c("#f4a91c","#955da2")) +
  theme_minimal(base_size = 14)

p1 




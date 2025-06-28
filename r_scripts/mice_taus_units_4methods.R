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


df <- read.csv("D:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\mice\\dataset\\cut_30min\\tau_long_4_methods_df.csv", 
               stringsAsFactors = TRUE)

# relevel
df$method <- relevel(df$method, ref = "acf_full")

# scale
df <- df %>%
  mutate(
    fr_s            = as.numeric(scale(fr_hz_spont_30min)),
    alpha_s         = as.numeric(scale(lv))
  )

# fit model

model_log <- lmer(
  log_tau_ms ~ method * (fr_s + alpha_s)
  + (1 | unit_id),
  data = df,
  REML = TRUE
)

summary(model_log)

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
    ratio     = (10**estimate - 1), # back‐transform
    ci.low    = (10**conf.low - 1),             
    ci.high   = (10**conf.high - 1),                
    term      = recode(term, # labels
                       "methodisttc_full" = "iSTTC",
                       "methodpearsonr_trial_avg" = "PearsonR",
                       "methodsttc_trial_concat" = "iSTTC concat",
                       "fr_s" = "Firing rate (SD)",
                       "alpha_s" = "Alpha (SD)",
                       "methodisttc_full:fr_s" = "methodisttc_full × FR",
                       "methodisttc_full:alpha_s" = "methodisttc_full × Alpha",
                       "methodpearsonr_trial_avg:fr_s" = "methodpearsonr_trial_avg × FR",
                       "methodpearsonr_trial_avg:alpha_s" = "methodpearsonr_trial_avg × Alpha",
                       "methodsttc_trial_concat:fr_s" = "methodsttc_trial_concat × FR",
                       "methodsttc_trial_concat:alpha_s" = "methodsttc_trial_concat × Alpha"),
    
    term = factor(term, levels = c(
      "iSTTC",
      "PearsonR",
      "methodsttc_trial_concat" = "iSTTC concat",
      "fr_s" = "Firing rate (SD)",
      "alpha_s" = "Alpha (SD)",
      "methodisttc_full × FR",
      "methodisttc_full × Alpha",
      "methodpearsonr_trial_avg × FR",
      "methodpearsonr_trial_avg × Alpha",
      "methodsttc_trial_concat × FR",
      "methodsttc_trial_concat × Alpha"
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

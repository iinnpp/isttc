library(sjPlot)
library(broom.mixed)    # for tidy()
library(dplyr)
library(tidyr)
library(emmeans)
library(lme4)
library(lmerTest)
library(ggplot2)
library(interactions)


df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\tau_combined_plot_df.csv", 
               stringsAsFactors = FALSE)
str(df)
summary(df)
head(df)

df$method <- factor(df$method, levels = c("acf_full", "isttc_full"))

###### Clean dataset ######

df_clean <- df %>% 
  # remove rows with NA and fows with fit_r_squared < 0 
  drop_na() %>% 
  filter(fit_r_squared >= 0)

# keep only unit_ids that have both methods
df_paired <- df_clean %>% 
  group_by(unit_id) %>% 
  filter(n_distinct(method) == 2) %>% 
  ungroup()

# check
df_paired %>% 
  summarise(
    n_units = n_distinct(unit_id),
    n_rows  = n()
  )

summary(df_paired)

##### Compare methods: tau_diff_rel_log ######

# Fixed effect of method; random intercept for each unit_id
model_tau_diff_rel_log <- lmer(
  tau_diff_rel_log ~ method + (1 | unit_id),
  data = df_paired,
  REML = TRUE
)
summary(model_tau_diff_rel_log)

# plot effect size (Cohen's d)
tidy_mod <- broom.mixed::tidy(model_tau_diff_rel_log, effects="fixed", conf.int=TRUE) %>%
  filter(term=="methodisttc_full")
resid_sd <- sigma(model_tau_diff_rel_log)

# get Cohen’s d–style standardized effect
std_eff <- tidy_mod %>%
  transmute(
    term        = term,
    est         = estimate,
    ci_low      = conf.low,
    ci_high     = conf.high,
    d           = estimate / resid_sd,
    d_ci_low    = conf.low / resid_sd,
    d_ci_high   = conf.high / resid_sd
  )

ggplot(std_eff, aes(x=term, y=d)) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin=d_ci_low, ymax=d_ci_high), width=0.2) +
  geom_hline(yintercept=0, linetype="dashed") +
  coord_flip() +
  scale_y_continuous(limits=c(-0.2, 0.2)) +
  labs(
    x = "", 
    y = "Standardized effect (Cohen’s d)",
    title = "Effect size of Method (isttc_full vs. acf_full)",
    subtitle = sprintf("d = %.3f (95%% CI [%.3f, %.3f])", std_eff$d, std_eff$d_ci_low, std_eff$d_ci_high)
  ) +
  theme_minimal(base_size=14)


##### Compare methods: fit_r_squared
model_fit_r_squared <- lmer(
  fit_r_squared ~ method + (1 | unit_id),
  data = df_paired,
  REML = TRUE
)
summary(model_fit_r_squared)

# plot effect size (Cohen's d)
tidy_mod <- broom.mixed::tidy(model_fit_r_squared, effects="fixed", conf.int=TRUE) %>%
  filter(term=="methodisttc_full")
resid_sd <- sigma(model_fit_r_squared)

# get Cohen’s d–style standardized effect
std_eff <- tidy_mod %>%
  transmute(
    term        = term,
    est         = estimate,
    ci_low      = conf.low,
    ci_high     = conf.high,
    d           = estimate / resid_sd,
    d_ci_low    = conf.low / resid_sd,
    d_ci_high   = conf.high / resid_sd
  )

ggplot(std_eff, aes(x=term, y=d)) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin=d_ci_low, ymax=d_ci_high), width=0.2) +
  geom_hline(yintercept=0, linetype="dashed") +
  coord_flip() +
  scale_y_continuous(limits=c(-0.2, 0.2)) +
  labs(
    x = "", 
    y = "Standardized effect (Cohen’s d)",
    title = "Effect size of Method (isttc_full vs. acf_full)",
    subtitle = sprintf("d = %.3f (95%% CI [%.3f, %.3f])", std_eff$d, std_eff$d_ci_low, std_eff$d_ci_high)
  ) +
  theme_minimal(base_size=14)


######## GIANT MODEL

# standardize predictors for interpretability:
df_paired <- df_paired %>%
  mutate(
    fr_c        = scale(fr,        center=TRUE, scale=TRUE)[,1],
    alpha_c     = scale(alpha,     center=TRUE, scale=TRUE)[,1],
    tau_ms_c    = scale(tau_ms_true, center=TRUE, scale=TRUE)[,1]
  )

# one big LME with all main effects + interactions with method
mod_all <- lmer(
  tau_diff_rel_log ~ 
    fr_c + alpha_c + tau_ms_c      # main effects
  + method                         # method difference at “average” predictor values
  + fr_c*method                   # does fr→tau_diff differ by method?
  + alpha_c*method                # does alpha→tau_diff differ?
  + tau_ms_c*method               # does tau_ms_true→tau_diff differ?
  + (1 | unit_id),                 # random intercept per unit
  data = df_paired,
  REML = TRUE
)

summary(mod_all)


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

##### ACF decline #####
# relevel
df$acf_decline <- relevel(df$acf_decline, ref = "False")

model_simple <- lmer(
  log_tau_diff ~ acf_decline + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_simple)


##### fit_r_squared #####
df <- df %>%
  mutate(
    log_fit_r_squared = log10(fit_r_squared)
  )

model_simple_r <- lmer(
  log_tau_diff ~ fit_r_squared + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_simple_r)

##### CI 0 #####
# relevel
df$ci_zero_excluded <- factor(df$ci_zero_excluded)
df$ci_zero_excluded <- relevel(df$ci_zero_excluded, ref = "0")

model_simple_ci <- lmer(
  log_tau_diff ~ ci_zero_excluded + (1 | unit_id),
  data = df,
  REML = TRUE
)
summary(model_simple_ci)



# violins
ggplot(df, aes(x = acf_decline, y = log_tau_diff, fill = acf_decline)) +
  geom_violin(trim = FALSE, scale = "width", alpha = 0.7) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.4) +
  stat_summary(fun = median, geom = "point", shape = 23, size = 2, fill = "white") +
  labs(
    title = "REE log10",
    x = "ACF Decline",
    y = "log10(tau_diff_rel)",
    fill = "ACF Decline"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    text = element_text(size = 14)
  )


# forest plot of fixed effects - simple version
p <- plot_model(
  model_simple, 
  show.values = TRUE,
  value.offset = .3,
  value.size = 4,
  dot.size = 2,
  line.size = 1,
  vline.color = "blue",
  width = 0.1
)
p10 <- p + scale_y_continuous(limits = c(-1, -0.75))
p10

# forest plot of fixed effects
fe <- tidy(model_simple, effects = "fixed", conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>% # drop the intercept
  mutate(
    ratio     = (10**estimate - 1)*100, # back‐transform
    ci.low    = (10**conf.low - 1)*100,             
    ci.high   = (10**conf.high - 1)*100,                
    term      = recode(term, # labels
                       "acf_declineTrue" = "acf_declineTrue"),
    
    term = factor(term, levels = c(
      "acf_declineTrue"
    )),
    term = factor(term, levels = rev(levels(factor(term)))))

ggplot(fe, aes(x = term, y = ratio)) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high), width = 0.2) +
  geom_point(size = 3, color = "steelblue") +
  coord_flip() +
  scale_y_continuous(
    name = "Effect on tau_diff_rel",
    limits = c(-90, -85)
  ) +
  labs(
    x     = NULL,
    title = "Fixed-Effects (Back-transformed from log10 to %)"
  ) + 
  theme_minimal(base_size = 14)




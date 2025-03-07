Sys.setenv(LANG = "en")
rm(list  =  ls())


#library('performance')
library('sjPlot')
library('lme4')
library('rio')
library('emmeans')
library('ggplot2')
#library('lmerTest')

df <- import("Q:\\Personal\\Irina\\projects\\isttc\\results\\monkey\\fixation_period_1000ms_no_empty\\acf_tau_full_df.csv")

df$method <- as.factor(df$method)
df$area <- as.factor(df$area)
df$unit_id <- as.factor(df$unit_id)

df_clean <- na.omit(df[df$decline_150_250 == TRUE, c("tau_ms", "method", "area", "unit_id", "tau_ms_log_10")])
df_clean$unique_id <- interaction(df_clean$area, df_clean$unit_id)

lmm <- lmer(tau_ms_log_10 ~ method + (1 | unique_id), data = df_clean)
summary(lmm)  
anova(lmm, type="III")  

emmeans(lmm, pairwise ~ method, adjust = "tukey")


# Get estimated marginal means (EMMs) for method
emm_results <- emmeans(lmm, ~ method)

# Convert to dataframe for ggplot
emm_df <- as.data.frame(emm_results)

# Plot estimated effects
ggplot(emm_df, aes(x = method, y = emmean, ymin = lower.CL, ymax = upper.CL)) +
  geom_point(size = 3, color = "black") +  # Mean points
  geom_errorbar(width = 0.2, color = "black") +  # Confidence intervals
  labs(x = "Method", y = "Estimated tau_ms (log scale)", 
       title = "Effect of Method on tau_ms") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


library(rstatix)  # For significance brackets

# Get estimated marginal means (EMMs) for method
emm_results <- emmeans(lmm, ~ method)

# Convert to dataframe for plotting
emm_df <- as.data.frame(emm_results)

# Pairwise comparisons with Tukey correction
pairs_results <- pairs(emm_results, adjust = "tukey")
pairs_df <- as.data.frame(pairs_results)

# Add significance labels based on p-values
pairs_df$significance <- cut(pairs_df$p.value, 
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf), 
                             labels = c("***", "**", "*", "ns"))

# Create significance brackets for ggplot
sig_brackets <- data.frame(
  x1 = pairs_df$contrast,  # First method
  x2 = pairs_df$contrast,  # Second method
  y = max(emm_df$emmean) + 0.2,  # Adjust height
  label = pairs_df$significance
)

# Plot estimated effects with significance
ggplot(emm_df, aes(x = method, y = emmean, ymin = lower.CL, ymax = upper.CL)) +
  geom_point(size = 3, color = "black") +  # Mean points
  geom_errorbar(width = 0.2, color = "black") +  # Confidence intervals
  labs(x = "Method", y = "Estimated tau_ms (log scale)", title = "Effect of Method on tau_ms") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_signif(comparisons = list(c("pearsonr_trial_avg", "STTC_trial_avg"),
                                 c("pearsonr_trial_avg", "STTC_trial_concat"),
                                 c("STTC_trial_avg", "STTC_trial_concat")),
              map_signif_level = TRUE, 
              test = "t.test")  # Adds significance brackets






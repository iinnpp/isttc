library(rio)
library(dabestr)
library(ggplot2)

df <- import("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\allen_mice\\dataset\\cut_30min\\summary_taus_plot_5methods_100_resampling_df_long.csv")
df$method <- factor(df$method, levels = c("acf_full", "isttc_full", "sttc_avg", "sttc_concat", "pearsonr"))

dabest_obj <- dabestr::load(
  data = df,
  x = method,
  y = tau_ms_log10,
  idx = c("acf_full", "isttc_full", "sttc_avg", "sttc_concat", "pearsonr")
)

dabest_obj <- dabestr::mean_diff(dabest_obj, paired = FALSE)

# Extract effect sizes
effect_df <- dabest_obj$mean_diff

# Inspect the structure
print(effect_df)

ggplot(effect_df, aes(x = comparison, y = difference)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    y = "Mean difference in log10(tau_ms) vs acf_full",
    x = NULL,
    title = "Effect sizes of tau_ms across methods"
  ) +
  theme_minimal()
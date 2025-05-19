library(sjPlot)
library(ggplot2)
library(moments)

df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\lv_df.csv")

model <- lm(lv ~ alpha + fr + tau_ms, data = df)
summary(model)
confint(model)

# check assumptions - the QQ plot has deviations at both ends but
# with 100 000 data points it is not a major issue
res  <- residuals(model)

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


# plot effects
p <- plot_model(
  model, 
  show.values   = TRUE,
  value.offset  = .3,
  value.size    = 4,
  dot.size      = 2,
  line.size     = 1,
  vline.color   = "blue",
  width         = 0.1
)
p1 <- p + scale_y_continuous(limits = c(-0.05, 0.45))
p1 + coord_flip()


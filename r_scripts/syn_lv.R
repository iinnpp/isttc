df <- read.csv("E:\\work\\q_backup_06_03_2025\\projects\\isttc\\results\\synthetic\\results\\param_fr_alpha_tau\\lv_df.csv")

model <- lm(lv ~ alpha + fr + tau_ms, data = df)
summary(model)

plot(model, which = 1)
plot(model, which = 2)
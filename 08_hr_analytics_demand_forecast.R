# Q8. HR analytics: predict demand for hourly employees for the following month (R)
# Approach: multiple linear regression on past hourly demand using
# hour-of-day, day-of-week and trend, then forecast every hour of next month.
# Run: Rscript 08_hr_analytics_demand_forecast.R
# Packages: base R only (uses stats::lm)

set.seed(42)

# 1. Simulate 6 months of hourly staffing demand (replace with your real CSV:
#    hr <- read.csv("hourly_demand.csv"), columns: datetime, employees_needed)
start <- as.POSIXct("2026-04-01 00:00:00", tz = "UTC")
hours <- seq(start, by = "hour", length.out = 24 * 183)   # Apr - Sep 2026
hr <- data.frame(datetime = hours)
hr$hour  <- as.integer(format(hr$datetime, "%H"))
hr$wday  <- factor(weekdays(hr$datetime),
                   levels = c("Monday", "Tuesday", "Wednesday", "Thursday",
                              "Friday", "Saturday", "Sunday"))
hr$trend <- seq_len(nrow(hr))

peak <- ifelse(hr$hour >= 9 & hr$hour <= 20, 1, 0)
hr$employees_needed <- round(
  8 + 25 * peak + 6 * (hr$hour %in% c(12, 13, 18, 19)) +
    5 * (hr$wday %in% c("Saturday", "Sunday")) +
    0.003 * hr$trend + rnorm(nrow(hr), 0, 2))
hr$employees_needed[hr$employees_needed < 0] <- 0

# 2. Train / test split by time (last 14 days = test)
n_test <- 24 * 14
train <- hr[1:(nrow(hr) - n_test), ]
test  <- hr[(nrow(hr) - n_test + 1):nrow(hr), ]

# 3. Model
model <- lm(employees_needed ~ factor(hour) + wday + trend, data = train)
print(summary(model)$r.squared)

# 4. Evaluate on the test period
pred <- predict(model, test)
rmse <- sqrt(mean((test$employees_needed - pred)^2))
mae  <- mean(abs(test$employees_needed - pred))
cat("RMSE:", round(rmse, 2), " MAE:", round(mae, 2), "\n")

# 5. Forecast the following month (October 2026), hour by hour
future <- data.frame(datetime = seq(as.POSIXct("2026-10-01 00:00:00", tz = "UTC"),
                                    by = "hour", length.out = 24 * 31))
future$hour  <- as.integer(format(future$datetime, "%H"))
future$wday  <- factor(weekdays(future$datetime), levels = levels(hr$wday))
future$trend <- max(hr$trend) + seq_len(nrow(future))
future$predicted_employees <- pmax(0, round(predict(model, future)))

cat("\nFirst 24 hours of forecast:\n")
print(head(future[, c("datetime", "predicted_employees")], 24))

# 6. Summary tables for HR planning
cat("\nAverage predicted demand by hour of day:\n")
print(round(tapply(future$predicted_employees, future$hour, mean), 1))
cat("\nAverage predicted demand by weekday:\n")
print(round(tapply(future$predicted_employees, future$wday, mean), 1))

write.csv(future, "hourly_demand_forecast_next_month.csv", row.names = FALSE)

png("hr_demand_forecast.png", width = 900, height = 450)
plot(future$datetime[1:(24 * 7)], future$predicted_employees[1:(24 * 7)],
     type = "l", col = "blue", xlab = "Time", ylab = "Employees needed",
     main = "Forecast: hourly employee demand (first week of next month)")
dev.off()
cat("\nSaved: hourly_demand_forecast_next_month.csv, hr_demand_forecast.png\n")

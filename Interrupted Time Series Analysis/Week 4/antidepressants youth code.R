setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 4")

library(nlme)
library(car)

########################
# Read in the dataset
########################

data <- read.csv("antidepressants_youth.csv", header = T)
View(data)

########################
# Initial Plot
########################

plot(data$time, data$ad_perc,
     ylab = "Antidepressant Percentage",
     ylim = c(1.0,2.0),
     xlab = "Quarter",
     type = "l",
     col = "red",
     xaxt = "n")

axis(1, at = 1:44, labels = data$quarter)

points(data$time, data$ad_perc,
       col = "red",
       pch = 20)

abline(v = 15.5, lty = 2)

#########################
# Create New Dataset
#########################

# Make a vector of the rows we want to include
include <- c(1:15, 21:44)

# Duplicate these rows into a new dataset
data_pi <- data[include,]

# Correct the trend variable in the new dataset
data_pi$trend[16:39] <- data_pi$trend[16:39] - 5

########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(ad_perc ~ time + level + trend, data = data_pi)
summary(model_ols)

################################
# Assessing Autocorrelation
################################

dwt(model_ols, max.lag = 8, alternative = "two.sided")

plot(data_pi$time,
     residuals(model_ols),
     type = 'o',
     pch = 16,
     xlab = 'Time',
     ylab = 'OLS Residuals',
     col = "red")
abline(h = 0,lty = 2)

par(mfrow = c(1,2))
acf(residuals(model_ols))
acf(residuals(model_ols),type = 'partial')
#significant spikes in ACF at lags 4,5 and 9 and significant spikes in PACF at lags 4,5 and 6
#model p = 5

par(mfrow = c(1,1))

########################
# Modeling
########################

# Fit the GLS regression model
model_p5 <- gls(ad_perc ~ time + level + trend,
  data = data_pi,
  correlation = corARMA(p = 5, form = ~time),
  method = "ML")
summary(model_p5)
confint(model_p5)

#estimate for the level change in the percentage of youths prescribed an antidepressant after the
#FDA warning
model_p5$coef[3]
#estimate for the trend change
model_p5$coef[4]

#running model without the phase-in period and compare the level change
model_p5_2 <- gls(ad_perc ~ time + level + trend,
                  data = data,
                  correlation = corARMA(p = 5, form = ~time),
                  method = "ML")
summary(model_p5_2)
#estimate with phase-in period minus estimate without phase-in period
model_p5$coef[3] - model_p5_2$coef[3]
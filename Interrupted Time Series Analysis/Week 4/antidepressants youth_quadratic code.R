setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 4")

library(nlme)
library(car)

########################
# Read in the dataset
########################

data <- read.csv("antidepressants youth_quadratic.csv", header = T)
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

rect(15.5, -5, 20.5, 20, border = NA, col = "#00000011")

#########################
# Create New Dataset
#########################

include <- c(1:15, 21:44)
data_pi <- data[include,]
data_pi$trend[16:39] <- data_pi$trend[16:39] - 5
data_pi$trendsq <- data_pi$trend^2

########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(ad_perc ~ time + level + trend + trendsq, data = data_pi)
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
#significant spikes in ACF at lags 4, 5 and 8, and significant spikes in PACF at lags 4,5 and 6
#model p = 5

par(mfrow = c(1,1))

########################
# Modeling
########################

# Fit the GLS regression model
model_p5 <- gls(ad_perc ~ time + level + trend + trendsq,
  data = data_pi,
  correlation = corARMA(p = 5, form = ~time),
  method = "ML")
summary(model_p5)
confint(model_p5)

########################
# Plot results
#########################

plot(data$time, data$ad_perc,
     ylab = "Antidepressant Percentage",
     ylim = c(1.0,3.0),
     xlab = "Quarter",
     pch = 20,
     col = "pink",
     xaxt = "n")

axis(1, at = 1:44, labels = data$quarter)

abline(v = 15.5, lty = 2)

rect(15.5, -5, 20.5, 20, border = NA, col = "#00000011")

#first line segment
lines(data$time[1:15], fitted(model_p5)[1:15], col = "red", lwd = 2)
#second line segment
lines(data$time[21:44], fitted(model_p5)[16:39], col="red", lwd = 2)
#counterfactual
segments(21, model_p5$coef[1] + model_p5$coef[2]*21,
         44, model_p5$coef[1] + model_p5$coef[2]*44,
         lty = 2, lwd = 2, col= 'red')

#estimate the absolute change at 2 years after the phase-in period i.e. 2006 Q4
pred <- fitted(model_p5)[23]
cfac <- model_p5$coef[1] + model_p5$coef[2]*28

pred - cfac

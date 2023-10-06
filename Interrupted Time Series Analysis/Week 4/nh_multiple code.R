setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 4")

library(nlme)
library(car)

data <- read.csv("nh_multiple.csv", header = T)
View(data)

data$antic <- rep(0,48)
data$antic[20] <- 1

data$cap <- c(rep(0,20), rep(1,28))
data$cap_trend <- c(rep(0,20), 1:28)
data$copay <- c(rep(0,31), rep(1,17))
data$copay_trend <- c(rep(0,31), 1:17)

View(data)

########################
# Initial Plot
########################

plot(data$time, data$rxpp,
       ylab = "Prescriptions per person",
       ylim = c(0,7),
       xlab = "Month",
       type = "l",
       col = "red",
       xaxt = "n")

axis(1, at = 1:48, labels = data$month)

points(data$time, data$rxpp,
       col = "red",
       pch = 20)

abline(v = 20.5,lty = 2)
abline(v = 31.5,lty = 2)

########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(rxpp ~ time + antic + cap + cap_trend + copay + copay_trend,
                data = data)
summary(model_ols)

################################
# Assessing Autocorrelation
################################

dwt(model_ols, max.lag = 12, alternative = "two.sided")

plot(residuals(model_ols),
     type = 'o',
     pch = 16,
     xlab = 'Time',
     ylab = 'OLS Residuals',
     col = "red")
abline(h = 0, lty = 2)

par(mfrow = c(1,2))
acf(residuals(model_ols))
acf(residuals(model_ols),type = 'partial')
#significant spike in ACF at lag 6 and lag 6 in PACF
#model (p,q) = (0,0)

par(mfrow = c(1,1))

########################
# Run the final model
########################

# Fit the GLS regression model
model_p0q0 <- gls(rxpp ~ time + antic + cap + cap_trend + copay + copay_trend,
  data = data,
  correlation = NULL,
  method = "ML")
summary(model_p0q0)
confint(model_p0q0)

#estimate for level change for the 3-drug-cap
model_p0q0$coef[4]
#estimate for the trend change for the 3-drug-cap
model_p0q0$coef[5]
#estimate for the level change for the change from the cap to the $1 copay
model_p0q0$coef[6]
#estimate for the trend change from the cap to the $1 copay
model_p0q0$coef[7]

##############################################
# Predict absolute and relative changes
##############################################

#predicted value at one year after the copay (July 1983) relative to the level and trend with the
#3-drug-cap
pred <- fitted(model_p0q0)[43]
#counterfactual at the same time point
cfac <- model_p0q0$coef[1] + model_p0q0$coef[2]*43 +
        model_p0q0$coef[4]*1 + model_p0q0$coef[5]*23
#absolute change
pred - cfac
#relative change
(pred - cfac) / cfac * 100

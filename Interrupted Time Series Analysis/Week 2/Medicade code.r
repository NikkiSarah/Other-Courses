setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 2")

library(nlme)
library(car)

data <- read.csv("antipsychotic_study.csv", header = T)
View(data)

########################
# Initial Plot
########################

plot(data$time, data$marketshare,
       ylab = "Market Share",
       ylim = c(0,0.6),
       xlab = "Year",
       type = "l",
       col = "red",
       xaxt = "n")

axis(1, at = 1:19, labels = data$yearqtr)

points(data$time, data$marketshare,
       col = "red",
       pch = 20)

abline(v = 8.5, lty = 2)

########################
# Modeling
########################

model_ols <- lm(marketshare ~ time + level + trend, data = data)
summary(model_ols)

#estimate of existing level
model_ols$coef[1]
#estimate of pre-existing trend per quarter
model_ols$coef[2]
#estimated level change following the prior authorisation policy
model_ols$coef[3]
#estimated change in the trend per quarter following the policy change
model_ols$coef[4]
#based on the standard model results, the policy in West Virginia lead to a level and trend decrease
#in market share

dwt(model_ols, max.lag = 8, alternative = "two.sided")
#the DW test statistic is significantly different from 2 on none of the specified lag values

plot(data$time,
     residuals(model_ols),
     type = 'o',
     pch = 16,
     xlab = 'Time',
     ylab = 'OLS Residuals',
     col = "red")
abline(h = 0, lty = 2)

par(mfrow = c(1,2))

acf(residuals(model_ols))
acf(residuals(model_ols), type = 'partial')
# Note no significant spikes in ACF or PACF. Model (p,q) = (0,0)

########################
# Run the final model
########################

#fit the GLS regression model
model_p0q0 <- gls(marketshare ~ time + level + trend,
  data = data,
  correlation = NULL,
  method = "ML")
summary(model_p0q0)

########################
# Diagnostic tests
########################

#likelihood-ratio tests to check AR/MA process
model_p1 <- update(model_p0q0, correlation = corARMA(p = 1,form = ~time))
anova(model_p0q0, model_p1)

model_q1 <- update(model_p0q0, correlation = corARMA(q = 1,form = ~time))
anova(model_p0q0,model_q1)

#residual plot
#null hypothesis: the residuals of a correctly specified model are independently distributed
#--the residuals are white noise
par(mfrow = c(1,1))
qqPlot(residuals(model_p0q0))

########################
# Plot results
#########################

#first plot the raw data points
plot(data$time, data$marketshare,
     ylim = c(0,0.6),
     ylab = "Market Share",
     xlab = "Year",
     pch = 20,
     col = "pink",
     xaxt = "n")

#add x axis with dates
axis(1, at = 1:19, labels = data$year)

#add line indicating policy change
abline(v = 8.5,lty = "dotted")

#plot the first line segment
lines(data$time[1:8], fitted(model_p0q0)[1:8], col = "red",lwd = 2)
#plot the second line segment
lines(data$time[9:19], fitted(model_p0q0)[9:19], col="red",lwd = 2)

#and the counterfactual
segments(1,
         model_p0q0$coef[1] + model_p0q0$coef[2],
         19,
         model_p0q0$coef[1]+model_p0q0$coef[2]*19,
         lty = 2,
         lwd = 2,
         col = 'red')

#estimate of predicted outcome at time point 15
pred0 <- model_p0q0$coef[1] + model_p0q0$coef[2]*15 + model_p0q0$coef[3] + model_p0q0$coef[4]*7
#estimate of counterfactual at same time point
cfac0 <- model_p0q0$coef[1] + model_p0q0$coef[2]*15
#absolute predicted change at that time point
pred0 - cfac0

##############################################
# Predict absolute and relative changes
##############################################

#predicted value at six quarters after the policy change
pred <- fitted(model_p0q0)[8 + 6]
#estimate the counterfactual at the same time point
cfac <- model_p0q0$coef[1] + model_p0q0$coef[2]*(8 + 6)
#absolute change at 6 quarters
pred - cfac
#relative change at 6 quarters
(pred - cfac) / cfac * 100

#predicted value at eight quarters after the policy change
pred <- fitted(model_p0q0)[8 + 8]
#estimate the counterfactual at the same time point
cfac <- model_p0q0$coef[1] + model_p0q0$coef[2]*(8 + 8)
#absolute change at 8 quarters
pred - cfac
#relative change at 8 quarters
(pred - cfac) / cfac * 100

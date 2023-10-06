setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 3")

library(nlme)
library(car)

########################
# Read in the dataset
########################

data <- read.csv("antipsychotic_study_control.csv", header = T)

data$WV <- c(rep(1,19), rep(0,19))
data$time <- c(1:19, 1:19)
data$level <- c(rep(0,8), rep(1,11), rep(0,8), rep(1,11))
data$trend <- c(rep(0,8), 1:11, rep(0,8), 1:11)
data$WVtime <- data$WV * data$time
data$WVlevel <- data$WV * data$level
data$WVtrend <- data$WV * data$trend

########################
# Initial Plot
########################

plot(data$time[1:19], data$market_share[1:19],
     ylab = "Market Share",
     ylim = c(0, 0.6),
     xlab = "Year",
     type = "l",
     col = "red",
     xaxt = "n")

points(data$time[20:38], data$market_share[20:38],
       type = 'l',
       col = "blue")

axis(1, at = 1:38, labels = data$yearqtr[1:38])

points(data$time[1:19], data$market_share[1:19],
       col = "red",
       pch = 20)

points(data$time[20:38], data$market_share[20:38],
       col = "blue",
       pch = 20)

abline(v = 8.5, lty = 2)

legend(x = 2, y = 0.2, legend = c("West Virginia", "Control States"),
       col = c("red", "blue"),pch = 20)
#if the control group is just shifted vertically slightly, the estimates for the level change and
#trend change would remain the same. Thus, the estimate of the counterfactual for the intervention
#group would not change, resulting in the same effect estimates.

########################
# Modeling
########################

model_ols <- lm(market_share ~ time + WV + WVtime + level + trend + WVlevel + WVtrend,
                data = data)
summary(model_ols)

#the coefficients representing the pre-existing level and trend differences between West Virginia and
#the control states, respectively are:
model_ols$coef[3]
model_ols$coef[4]

#the coefficients representing the level and trend changes in the control states respectively, are:
model_ols$coef[5]
model_ols$coef[6]

#the estimated level change following the prior authorisation policy relative to the control group
model_ols$coef[7]

#the estimated change in the trend per quarter following the policy change relative to the
#control group
model_ols$coef[8]

#based on the standard model results with a control group, the policy in West Virginia lead to a
#level decrease and trend decrease in market share

################################
# Assessing Autocorrelation
################################

dwt(model_ols, max.lag = 8, alternative = "two.sided")
#the Durbin Watson test statistic is significantly different from 2 on the 1st lag value

#graph the residuals from the OLS regression to check for serially correlated errors
plot(data$time[1:19],
	residuals(model_ols)[1:19],
	type = 'o',
	pch = 16,
	xlab = 'Time',
	ylab = 'OLS Residuals',
	col = "red")
abline(h = 0, lty = 2)

#plot ACF and PACF
par(mfrow = c(1,2))

acf(residuals(model_ols))
acf(residuals(model_ols),type = 'partial')
#note nothing major in plots, DWT = 1, model p = 1

########################
# Run the final model
########################
  
# Fit the GLS regression model
model_p1 <- gls(market_share ~ time + WV + WVtime + level + trend + WVlevel + WVtrend,
                data = data,
                correlation = corARMA(p = 1, form = ~time|WV),
                method="ML")
summary(model_p1)
confint(model_p1)

#expected level in the control states
cont <- model_p1$coef[1] + model_p1$coef[2]*5
#expected level in West Virginia
WV <- model_p1$coef[1] + model_p1$coef[2]*5 + model_p1$coef[3]*1 + model_p1$coef[4]*5
WV - cont
#at time point 5, the market share in West Virginia was predicted to be 0.021 higher than the control
#states

#estimate of the level change in West Virginia relative to the control states
model_p1$coef[7]
#lower bound on this estimate
-0.05635
#upper bound on this estimate
-0.01387

#estimate of the trend change in West Virginia relative to the control states
model_p1$coef[8]
#lower bound on this estimate
-0.01735
#upper bound on this estimate
-0.00843

########################
# Diagnostic tests
########################
  
# Likelihood-ratio tests to check whether the parameters of the AR process for the errors are necessary and sufficient
model_p6 <- update(model_p1, correlation = corARMA(p = 6, form = ~time|WV))
anova(model_p1, model_p6)

model_p2 <- update(model_p1, correlation = corARMA(p = 2, form = ~time|WV))
anova(model_p1, model_p2)

model_p1q1 <- update(model_p1, correlation = corARMA(p = 1, q = 1, form = ~time|WV))
anova(model_p1, model_p1q1)

# Put plotting back to one chart
par(mfrow=c(1,1))

# Residual plot
qqPlot(residuals(model_p1))


########################
# Plot results
#########################

plot(data$time[1:19], data$market_share[1:19],
          ylim  =  c(0,0.6),
          ylab = "Market share of non-preferred agents",
          xlab = "Quarter",
          pch = 20,
          col = "lightblue",
          xaxt = "n")

axis(1, at = 1:19, labels = data$yearqtr[1:19])

abline(v = 8.5, lty = 2)

points(data$time[20:38], data$market_share[20:38],
       col = "pink",
       pch = 20)

#first line segment for the intervention group
lines(data$time[1:8], fitted(model_p1)[1:8], col = "blue", lwd = 2)
#second line segment for the intervention group
lines(data$time[9:19], fitted(model_p1)[9:19], col = "blue", lwd = 2)
#counterfactual for the intervention group
segments(9, model_p1$coef[1] + model_p1$coef[2]*9 + model_p1$coef[3] + model_p1$coef[4]*9 + 
           model_p1$coef[5] + model_p1$coef[6],
         19, model_p1$coef[1] + model_p1$coef[2]*19 + model_p1$coef[3] + model_p1$coef[4]*19 + 
           model_p1$coef[5] + model_p1$coef[6]*11,
         lty = 2, col = 'blue', lwd = 2)

#first line segment for the control group
lines(data$time[20:27], fitted(model_p1)[20:27], col = "red",lwd = 2)
#second line segment for the control
lines(data$time[28:38], fitted(model_p1)[28:38], col = "red",lwd = 2)
#counterfactual for the control group
segments(9, model_p1$coef[1] + model_p1$coef[2]*9,
         19, model_p1$coef[1] + model_p1$coef[2]*19,
         lty = 2,col = 'red',lwd = 2)

legend(x = "bottomleft", legend = c("West Vriginia", "Control States"), col = c("blue","red"),
       pch = 20)

##############################################
# Predict absolute and relative changes
##############################################

#predicted value at 6 quarters after the policy was introduced
pred <- fitted(model_p1)[14]
#counterfactual at the same time point
cfac <- model_p1$coef[1] + model_p1$coef[2]*14 +
        model_p1$coef[3] + model_p1$coef[4]*14 +
        model_p1$coef[5] + model_p1$coef[6]*6
# Absolute change at 6 quarters
pred - cfac
# Relative change at 6 quarters
(pred - cfac) / cfac * 100

#predicted value at 8 quarters after the policy was introduced
pred <- fitted(model_p1)[16]
#counterfactual at the same time point
cfac <- model_p1$coef[1] + model_p1$coef[2]*16 +
  model_p1$coef[3] + model_p1$coef[4]*16 +
  model_p1$coef[5] + model_p1$coef[6]*8
# Absolute change at 8 quarters
pred - cfac
# Relative change at 8 quarters
(pred - cfac) / cfac * 100
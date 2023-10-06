#####################################
# ITSx: Session 2 Example Analysis
# Michael Law (michael.law@ubc.ca)
# July 2015
#####################################

###################################
# Load the necessary libraries
###################################

library(nlme)
library(car)


########################
# Read in the dataset
########################

# Two options: either use "import dataset" in RStudio, or use read.csv
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nile.csv",header=T)
# Examine the dataset within RStudio
View(data)


########################
# Initial Plot
########################

# Plot outcome variable versus time
plot(data$time,data$flow,
       ylab="Water Flow of the Nile",
       ylim=c(0,4500),
       xlab="Year",
       type="l",
       col="red",
       xaxt="n")

# Add x-axis year labels
axis(1, at=1:60, labels=data$year)

# Add in the points for the figure
points(data$time,data$flow,
       col="red",
       pch=20)

# Label the weather change
abline(v=27.5,lty=2)


########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(flow ~ time + level + trend, data=data)
summary(model_ols)

# Durbin-watson test, 12 time periods
dwt(model_ols,max.lag=12,alternative="two.sided")

# Graph the residuals from the OLS regression to check for serially correlated errors
plot(data$time,
     residuals(model_ols),
     type='o',
     pch=16,
     xlab='Time',
     ylab='OLS Residuals',
     col="red")
abline(h=0,lty=2)

# Plot ACF and PACF
# Set plotting to two records on one page
par(mfrow=c(1,2))

# Produce plots
acf(residuals(model_ols))
acf(residuals(model_ols),type='partial')
# Note decay in ACF, significant spike at 10 in PACF, model p=10


########################
# Run the final model
########################

# Fit the GLS regression model
model_p10 <- gls(flow ~ time + level + trend,
  data=data,
  correlation=corARMA(p=10,form=~time),
  method="ML")
summary(model_p10)


########################
# Diagnostic tests
########################

# Likelihood-ratio tests to check AR process
model_p11 <- update(model_p10,correlation=corARMA(p=11,form=~time))
anova(model_p10,model_p11)

model_p10q1 <- update(model_p10,correlation=corARMA(p=10,q=1,form=~time))
anova(model_p10,model_p10q1)

model_p10q10 <- update(model_p10,correlation=corARMA(p=10,q=10,form=~time))
anova(model_p10,model_p10q10)

# Residual plot
# Null Hypo: the residuals of a correctly specified model are independently distributed--the residuals are white noise
par(mfrow=c(1,1))
qqPlot(residuals(model_p10))


########################
# Plot results
#########################

# Produce the plot, first plotting the raw data points
plot(data$time,data$flow,
     ylim=c(0,4500),
     ylab="Water Flow",
     xlab="Year",
     pch=20,
     col="pink",
     xaxt="n")

# Add x axis with dates
axis(1, at=1:60, labels=data$year)

# Add line indicating weather pattern change
abline(v=27.5,lty="dotted")

# Plot the first line segment
lines(data$time[1:27], fitted(model_p10)[1:27], col="red",lwd=2)
# Plot the second line segment
lines(data$time[28:60], fitted(model_p10)[28:60], col="red",lwd=2)

# And the counterfactual
segments(1,
         model_p10$coef[1]+model_p10$coef[2],
         60,
         model_p10$coef[1]+model_p10$coef[2]*60,
         lty=2,
         lwd=2,
         col='red')


##############################################
# Predict absolute and relative changes
##############################################

# Predicted value at 25 years after the weather change
pred <- fitted(model_p10)[52]

# Then estimate the counterfactual at the same time point
cfac <- model_p10$coef[1] + model_p10$coef[2]*52

# Absolute change at 25 years
pred - cfac
# Relative change at 25 years
(pred - cfac) / cfac

# END
###########################################
# ITSx: Week 4 Multiple Interventions
# Michael Law (michael.law@ubc.ca)
# October 2015
###########################################

###################################
# Load the necessary libraries
###################################

library(nlme)
library(car)


########################
# Read in the dataset
########################

# Two options: either use "import dataset" in RStudio, or use read.csv
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/teststrips_multiple.csv",header=T)


########################
# Initial Plot
########################

# Plot outcome variable versus time
plot(data$time,data$strips_pt,
       ylab="Test Strips per 1,000 people",
       ylim=c(0,300),
       xlab="Month",
       type="l",
       col="red",
       xaxt="n")

# Add x-axis year labels
axis(1, at=1:72, labels=data$yearmonth)

# Add in the points for the figure
points(data$time,data$strips_pt,
       col="red",
       pch=20)

# Label the policy changes
abline(v=30.5,lty=2)
abline(v=56.5,lty=2)


########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(strips_pt ~ time + cerc + cerc_trend + cda + cda_trend, data=data)
summary(model_ols)


################################
# Assessing Autocorrelation
################################

# Durbin-watson test, 12 time periods
dwt(model_ols,max.lag=12,alternative="two.sided")

# Graph the residuals from the OLS regression to check for serially correlated errors
plot(residuals(model_ols),
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
# Spike in PACF at 4, exp decay in ACF, model with p=4

# Reset plot window
par(mfrow=c(1,1))

########################
# Run the final model
########################

# Fit the GLS regression model
model_p4 <- gls(strips_pt ~ time + cerc + cerc_trend + cda + cda_trend,
  data=data,
  correlation=corARMA(p=4,form=~time),
  method="ML")
summary(model_p4)


########################
# Plot results
#########################

# Produce the plot, first plotting the raw data points
plot(data$time,data$strips_pt,
     ylab="Test Strips per 1,000 people",
     ylim=c(0,300),
     xlab="Month",
     pch=20,
     col="pink",
     xaxt="n")

# Add x axis with dates
axis(1, at=1:72, labels=data$yearmonth)

# Add line indicating the policy changes
abline(v=30.5,lty=2)
abline(v=56.5,lty=2)

# Plot the first line segment
lines(data$time[1:30], fitted(model_p4)[1:30], col="red",lwd=2)

# Plot the second line segment
lines(data$time[31:56], fitted(model_p4)[31:56], col="red",lwd=2)

# Plot the third line segment
lines(data$time[57:72], fitted(model_p4)[57:72], col="red",lwd=2)

# And the first counterfactual
segments(31, model_p4$coef[1]+model_p4$coef[2]*31,
         56, model_p4$coef[1]+model_p4$coef[2]*56,
         lty=2, lwd=2, col='red')

# And the second counterfactual
segments(57, model_p4$coef[1] + model_p4$coef[2]*57 +
           model_p4$coef[3] + model_p4$coef[4]*27,
         72, model_p4$coef[1] + model_p4$coef[2]*72 +
           model_p4$coef[3] + model_p4$coef[4]*42,
         lty=2, lwd=2, col='red')

# END
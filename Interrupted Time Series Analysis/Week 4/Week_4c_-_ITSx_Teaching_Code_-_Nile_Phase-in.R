################################################
# ITSx: Week 4: Phase-in Periods
# Michael Law (michael.law@ubc.ca)
# October 2015
################################################

###################################
# Load the necessary libraries
###################################

library(nlme)
library(car)


########################
# Read in the dataset
########################

# Two options: either use "import dataset" in RStudio, or use read.csv
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nile_phase_in.csv",header=T)


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
axis(1, at=1:35, labels=data$year)

# Add in the points for the figure
points(data$time,data$flow,
       col="red",
       pch=20)

# Label the upstream dam build
abline(v=14.5,lty=2)


#########################
# Create New Dataset
#########################

# Make a vector of the rows we want to include
include <- c(1:14,19:35)

# Duplicate these rows into a new dataset
data_pi <- data[include,]

# Correct the trend variable in the new dataset
data_pi$trend[15:31] <- data_pi$trend[15:31] - 4


########################
# Modeling
########################

# A preliminary OLS regression
model_ols <- lm(flow ~ time + level + trend, data=data_pi)
summary(model_ols)


################################
# Assessing Autocorrelation
################################

# Durbin-watson test, 12 time periods
dwt(model_ols,max.lag=12,alternative="two.sided")

# Graph the residuals from the OLS regression to check for serially correlated errors
plot(data_pi$time,
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
# Note decay in ACF, significant spike at 4 in PACF, model p=4

# Reset the plot
par(mfrow=c(1,1))


########################
# Modeling
########################

# Fit the GLS regression model
model_p4 <- gls(flow ~ time + level + trend,
  data=data_pi,
  correlation=corARMA(p=4,form=~time),
  method="ML")
summary(model_p4)


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
axis(1, at=1:35, labels=data$year)

# Add line indicating upstream dam
abline(v=14.5,lty=2)

# Plot the first line segment
lines(data$time[1:14], fitted(model_p4)[1:14], col="red",lwd=2)

# Plot the second line segment
lines(data$time[19:35], fitted(model_p4)[15:31], col="red",lwd=2)

# And the counterfactual
segments(19, model_p4$coef[1]+model_p4$coef[2]*19,
         35, model_p4$coef[1]+model_p4$coef[2]*35,
         lty=2, lwd=2, col='red')

# Add a box to show phase-in period
rect(14.5,-500,18.5,5000 , border = NA, col= '#00000011')

# END
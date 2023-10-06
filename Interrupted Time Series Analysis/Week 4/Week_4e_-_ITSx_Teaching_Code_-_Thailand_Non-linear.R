################################################
# ITSx: Week 4: Non-linear Trends
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
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/thailand.csv",header=T)


########################
# Initial Plot
########################

# Plot outcome variable versus time
plot(data$time,data$stdunits,
     ylab="Standard Units of Insulin per 1,000 population",
     ylim=c(0,14),
     xlab="Quarter",
     type="l",
     col="red",
     xaxt="n")

# Add x-axis year labels
axis(1, at=1:34, labels=data$yearqtr)

# Add in the points for the figure
points(data$time,data$stdunits,
       col="red",
       pch=20)

# Label the policy change
abline(v=12.5,lty=2)

# Plot the phase-in period
rect(12.5,-5,15.5,20 , border = NA, col= '#00000011')


###########################################
# Create New Dataset to include phase-in
###########################################

# Make a vector of the rows we want to include
include <- c(1:12,16:34)

# Duplicate these rows into a new dataset
data_pi <- data[include,]

# Correct the trend and square variable in the new dataset
data_pi$trend[13:31] <- data_pi$trend[13:31] - 3
data_pi$trendsq <- data_pi$trend^2


#############################
# Modeling - with square term
#############################

# A preliminary OLS regression
model_ols <- lm(stdunits ~ time + level + trend + trendsq, data=data_pi)
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
# No significant autocorrelation, model p=0, q=0

# Reset the plot
par(mfrow=c(1,1))


########################
# Modeling
########################

# Fit the GLS regression model with square term
model_sq <- gls(stdunits ~ time + level + trend + trendsq,
                data=data_pi,
                method="ML")
summary(model_sq)


########################
# Plot results
#########################

# Produce the plot, first plotting the raw data points
plot(data$time,data$stdunits,
     ylim=c(0,14),
     ylab="Standard Units of Insulin per 1,000 population",
     xlab="Quarter",
     pch=20,
     col="pink",
     xaxt="n")

# Add x axis with dates
axis(1, at=1:34, labels=data$yearqtr)

# Add line indicating upstream dam
abline(v=12.5,lty=2)

# Plot the first line segment
lines(data$time[1:12], fitted(model_sq)[1:12], col="red",lwd=2)

# Plot the second line segment
lines(data$time[16:34], fitted(model_sq)[13:31], col="red",lwd=2)

# And the counterfactual
segments(16, model_sq$coef[1] + model_sq$coef[2]*16,
         34, model_sq$coef[1] + model_sq$coef[2]*34,
         lty=2, lwd=2, col='red')

# Add a box to show phase-in period
rect(12.5,-5,15.5,20 , border = NA, col= '#00000011')

# END
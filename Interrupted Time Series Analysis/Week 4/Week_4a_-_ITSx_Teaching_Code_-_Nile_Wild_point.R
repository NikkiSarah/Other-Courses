################################################
# ITSx: Week 4: Wild Points
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
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nile_week_4a.csv",header=T)

# Alternative method to create a new variable indicating the wild point
data$drought <- rep(0,60)
data$drought[43] <- 1


########################
# Preliminary Analysis
########################

# Fit the OLS regression model
model_ols <- lm(flow ~ time + level + trend + drought, data=data)
summary(model_ols)


########################
# Modeling
########################

# Fit the GLS regression model with p=10 as in Week 2
model_p10 <- gls(flow ~ time + level + trend + drought,
  data=data,
  correlation=corARMA(p=10,form=~time),
  method="ML")
summary(model_p10)


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

# Plot the second line segment - Note what happens!
lines(data$time[28:60], fitted(model_p10)[28:60], col="red",lwd=2)

# An alternative using model coefficients
segments(28,
         model_p10$coef[1] + model_p10$coef[2]*28 +
         model_p10$coef[3] + model_p10$coef[4],
         60,
         model_p10$coef[1] + model_p10$coef[2]*60 +
         model_p10$coef[3] + model_p10$coef[4]*33,
         lty=1,
         lwd=2,
         col='red')

# And the counterfactual
segments(1,
         model_p10$coef[1]+model_p10$coef[2],
         60,
         model_p10$coef[1]+model_p10$coef[2]*60,
         lty=2,
         lwd=2,
         col='red')

# END
################################################
# ITSx: Week 4: Seasonality
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
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nile_week_4b.csv",header=T)

# Alternative method to create a new variable indicating the wild point
data$drought <- rep(0,60)
data$drought[43] <- 1


########################
# Modeling
########################

# Fit the GLS regression model with p=10 as in Week 2
model_p10 <- gls(flow ~ time + level + trend + drought + elnino,
  data=data,
  correlation=corARMA(p=10,form=~time),
  method="ML")
summary(model_p10)


###############################
# Plot results - Approach 1
###############################

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


###############################
# Plot results - Approach 2
###############################

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

# Calculate the offset due to El Nino events
offset <- mean(data$elnino) * model_p10$coef[6]

# Plot the first line segment
segments(1,
         model_p10$coef[1] + model_p10$coef[2] + offset,
         27,
         model_p10$coef[1] + model_p10$coef[2]*27 + offset,
         lty=1, lwd=2, col='red')

# Plot the second line segment
segments(28,
         model_p10$coef[1] + model_p10$coef[2]*28 + 
           model_p10$coef[3] + model_p10$coef[4] + offset,
         60,
         model_p10$coef[1] + model_p10$coef[2]*60 + 
           model_p10$coef[3] + model_p10$coef[4]*33 + offset,
         lty=1, lwd=2, col='red')

# Plot the counterfactual
segments(1,
         model_p10$coef[1]+model_p10$coef[2] + offset,
         60,
         model_p10$coef[1]+model_p10$coef[2]*60 + offset,
         lty=2, lwd=2, col='red')

# END
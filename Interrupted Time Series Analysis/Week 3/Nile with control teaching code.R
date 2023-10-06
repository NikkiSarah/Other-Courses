#####################################
# ITSx: Session 3 Example Analysis
# Michael Law (michael.law@ubc.ca)
# October 2015
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
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nilehuron.csv",header=T)

# Setup variables
data$time <- c(1:60,1:60)
data$level <- c(rep(0,27),rep(1,33),rep(0,27),rep(1,33))
data$trend <- c(rep(0,27),1:33,rep(0,27),1:33)
data$niletime <- data$nile * data$time
data$nilelevel <- data$nile * data$level
data$niletrend <- data$nile * data$trend


########################
# Initial Plot
########################

# Plot the time series for the Nile at Aswan
plot(data$time[1:60],data$flow[1:60],
     ylab="Water Flow",
     ylim=c(0,4500),
     xlab="Year",
     type="l",
     col="red",
     xaxt="n")

# Add in control group flow into Lake Huron
points(data$time[61:120],data$flow[61:120],
       type='l',
       col="blue")

# Add x-axis year labels
axis(1, at=1:60, labels=data$year[1:60])

# Add in the points for the figure
points(data$time[1:60],data$flow[1:60],
       col="red",
       pch=20)

points(data$time[61:120],data$flow[61:120],
       col="blue",
       pch=20)

# Label the weather change
abline(v=27.5,lty=2)

# Add in a legend
legend(x=3, y=1000, legend=c("Nile","Huron"),
       col=c("red","blue"),pch=20)


########################
# Modeling
########################

# A preliminary OLS regression
model_ols<-lm(flow ~ time + nile + niletime + level + trend + nilelevel + 
          niletrend, data=data)
summary(model_ols)
confint(model_ols)


################################
# Assessing Autocorrelation
################################

# Durbin-watson test to 12 lags
dwt(model_ols,max.lag=12,alternative="two.sided")

# Graph the residuals from the OLS regression to check for serially correlated errors
plot(data$time[1:60],
	residuals(model_ols)[1:60],
	type='o',
	pch=16,
	xlab='Time',
	ylab='OLS Residuals',
	col="red")
abline(h=0,lty=2)

# Plot ACF and PACF
# Set plotting to two records on one page
par(mfrow=c(1,2))

# Produce Plots
acf(residuals(model_ols))
acf(residuals(model_ols),type='partial')
# Note decay in ACF, significant spike at 10 in PACF, model p=10


########################
# Run the final model
########################
  
# Fit the GLS regression model
model_p10 <- gls(flow ~ time + nile + niletime + level + trend + nilelevel + 
             niletrend,
             data=data,
             correlation=corARMA(p=10,form=~time|nile),
             method="ML")
summary(model_p10)
confint(model_p10)


########################
# Diagnostic tests
########################
  
# Likelihood-ratio tests to check whether the parameters of the AR process for the errors are necessary and sufficient
model_p10q1 <- update(model_p10,correlation=corARMA(q=1,p=10,form=~time|nile))
anova(model_p10,model_p10q1)
  
model_p11 <- update(model_p10,correlation=corARMA(p=11,form=~time|nile))
anova(model_p10,model_p11)

# Put plotting back to one chart
par(mfrow=c(1,1))

# Residual plot
qqPlot(residuals(model_p10))


########################
# Plot results
#########################

# First plot the raw data points for the Nile
plot(data$time[1:60],data$flow[1:60],
          ylim=c(0,4500),
          ylab="Water Flow",
          xlab="Year",
          pch=20,
          col="lightblue",
          xaxt="n")

# Add x-axis year labels
axis(1, at=1:60, labels=data$year[1:60])
# Label the policy change
abline(v=27.5,lty=2)

# Add in the points for the control
points(data$time[61:120],data$flow[61:120],
       col="pink",
       pch=20)

# Plot the first line segment for the intervention group
lines(data$time[1:27], fitted(model_p10)[1:27], col="blue",lwd=2)

# Add the second line segment for the intervention group
lines(data$time[28:60], fitted(model_p10)[28:60], col="blue",lwd=2)

# Add the counterfactual for the intervention group
segments(28, model_p10$coef[1] + model_p10$coef[2]*28 + model_p10$coef[3]+model_p10$coef[4]*28 + 
           model_p10$coef[5] + model_p10$coef[6],
         60, model_p10$coef[1] + model_p10$coef[2]*60 + model_p10$coef[3]+model_p10$coef[4]*60 + 
           model_p10$coef[5] + model_p10$coef[6]*33,
         lty=2,col='blue',lwd=2)

# Plot the first line segment for the control group
lines(data$time[61:87], fitted(model_p10)[61:87], col="red",lwd=2)

# Add the second line segment for the control
lines(data$time[88:120], fitted(model_p10)[88:120], col="red",lwd=2)

# Add the counterfactual for the control group
segments(1, model_p10$coef[1]+model_p10$coef[2],
         60,model_p10$coef[1]+model_p10$coef[2]*60,
         lty=2,col='red',lwd=2)

# Add in a legend
legend(x=3, y=1000, legend=c("Nile","Huron"), col=c("blue","red"),pch=20)


##############################################
# Predict absolute and relative changes
##############################################

# Predicted value at 25 years after the weather change
pred <- fitted(model_p10)[52]

# Estimate the counterfactual at the same time point
cfac <- model_p10$coef[1] + model_p10$coef[2]*52 +
        model_p10$coef[3] + model_p10$coef[4]*52 +
        model_p10$coef[5] + model_p10$coef[6]*25

# Absolute change at 25 years
pred - cfac
# Relative change at 25 years
(pred - cfac) / cfac

# END
setwd("//prod.protected.ind/user/User03/nf8331/desktop/Interrupted Time Series Analysis/Week 4")

library(nlme)
library(car)

########################
# Read in the dataset
########################

data <- read.csv("nh_wild_point.csv",header = T)

# Create a new variable indicating the wild point
data$antic <- rep(0,31)
data$antic[20] <- 1

########################
# Preliminary Analysis
########################

model_ols <- lm(rxpp ~ time + level + trend + antic, data = data)
summary(model_ols)

########################
# Modeling
########################

model <- gls(rxpp ~ time + level + trend + antic,
  data = data,
  correlation = NULL,
  method = "ML")
summary(model)

#model estimate without accounting for wild point
model_2 <- gls(rxpp ~ time + level + trend,
             data = data,
             correlation = NULL,
             method = "ML")
summary(model_2)

#calculating level change difference between models
model$coef[3] - model_2$coef[3]
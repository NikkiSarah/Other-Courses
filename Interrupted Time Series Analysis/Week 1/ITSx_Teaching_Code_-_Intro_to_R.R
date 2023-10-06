#####################################
# ITSx: Session 1 Introduction to R
# Michael Law (michael.law@ubc.ca)
# September 2015
#####################################


########################
# The Basics
########################

# This is a comment. RStudio shows them in green by default.

# Get help on a command
help(plot)

# install a new library
install.packages("car")

# Load a library
library(nlme)
library(car)

# Assigning values to an object
# R uses '<-' for this purpose
# For example, to make an object with value 3 called x:
x <- 3

# If you want R to print something, just type the name
x

# Or you can print it after an operation
x * 5

# You can also create vectors using 'c' to concatenate:
y <- c(1,3,5,2)
y

# To refer to a particular item in a vector, use square brackets:
y[3]


########################
# Datasets
########################

# Read in a CSV file
# You can also do this in RStudio using the "Import Dataset" button
data <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/nile_week_1.csv",header=T)

# Print the dataset
data

# See the variable names
names(data)

# Print the entire flow variable
data$flow

# Print just the third flow value
data$flow[3]

# Find the minimum and maximum values in the variable
min(data$flow)
max(data$flow)

# You can create a new variable using the assignment operator
# The data is in cubic metres, so make a new one in cubic feet
data$flowfeet <- data$flow * 35.3147


########################
# Basic Plotting
########################

plot(data$time,data$flow,
     ylab="Water Flow in Cubic Meters",
     xlab="Year",
     type="l",
     col="red")


########################
# Linear Regression
########################

# Run a basic linear model
ols.model <- lm(flow ~ time + drought, data=data)

# Different functions to work with the model
# Print a summary of the results
summary(ols.model)

# Produce confidence intervals
confint(ols.model)

# Get a vector of the coefficients
coef(ols.model)

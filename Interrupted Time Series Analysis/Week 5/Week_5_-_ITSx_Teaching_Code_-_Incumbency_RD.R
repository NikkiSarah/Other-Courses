################################################
# ITSx: Week 5: Regression Discontinuities
# Michael Law (michael.law@ubc.ca)
# October 2015
################################################

###################################
# Read in the data
###################################

# Read in the dataset
dataset <- read.csv("/Users/michaellaw/Dropbox/edX/Course Materials/Course Datasets/electoral_rd.csv",header=T)

# Constuct Variables
# dmargin = democratic margin of victory <- forcing variable
# demwin = democratic win <- indicator for threshold
# dwinnext = democratic win in next election <- outcome

# Setup square and cubic terms for forcing variable
dataset$dmargin2 <- dataset$dmargin^2
dataset$dmargin3 <- dataset$dmargin^3

# Setup interaction between forcing variable and threshold
dataset$dmargin_demwin <- dataset$dmargin * dataset$demwin

# Setup square and cubic terms for forcing variable * threshold interactions
dataset$dmargin_demwin2 <- dataset$dmargin_demwin^2
dataset$dmargin_demwin3 <- dataset$dmargin_demwin^3

# Set scientific notation to be off
options(scipen=5)


###################################
# Preliminary Plot
###################################

# Setup bins for plotting
bins <- seq(-49,49,2)

# Get a vector of the mean within each bin
means <- tapply(dataset$dwinnext,dataset$bin,mean)

# Plot the results
plot(bins,means,
     pch=19,
     ylab="Probability of Winning Next Election",
     xlab="Vote Margin in the Last Election",
     xlim=c(-50,50),
     col="lightblue")

# Add line at zero
abline(v=0,lty=2,col="grey")

###################################
# Modeling
###################################

# Run basic model
model1 <- lm(dwinnext ~ dmargin + demwin + dmargin_demwin,
            data=dataset)
summary(model1)

# Add square terms
model2 <- lm(dwinnext ~ dmargin + dmargin2 + 
             demwin + dmargin_demwin + dmargin_demwin2,
             data=dataset)
summary(model2)

# Compare versus model 1
anova(model1, model2)

# Run full specified model
model3 <- lm(dwinnext ~ dmargin + dmargin2 + dmargin3 + demwin
            + dmargin_demwin + dmargin_demwin2 + dmargin_demwin3,
            data=dataset)
summary(model3)

# Compare versus model 2
anova(model2, model3)


###################################
# Plot the Final Results
###################################

# Setup fake data frame for plotting
new <- data.frame(dmargin = seq(-50, 50, 0.5),
                  demwin=c(rep(0,100),rep(1,101)))
new$dmargin2 <- new$dmargin^2
new$dmargin3 <- new$dmargin^3
new$dmargin_demwin <- new$dmargin*new$demwin
new$dmargin_demwin2 <- new$dmargin_demwin^2
new$dmargin_demwin3 <- new$dmargin_demwin^3

# Plot the results
plot(bins,means,
     pch=19,
     ylab="Probability of Winning Next Election",
     xlab="Vote Margin in the Last Election",
     xlim=c(-50,50),
     ylim=c(0,1),
     col="lightblue")

# Add predicated values, use synthetic dataset
preds <- predict(model3, new)
lines(new$dmargin[1:100],preds[1:100],lwd=2,col="blue") 
lines(new$dmargin[102:200],preds[102:200],lwd=2,col="blue")

# Add line at zero
abline(v=0,lty=2,col="grey")


# END